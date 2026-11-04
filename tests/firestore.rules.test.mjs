import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { after, before, beforeEach, test } from 'node:test';
import { initializeTestEnvironment, assertFails, assertSucceeds } from '@firebase/rules-unit-testing';
import { collection, deleteDoc, doc, getDoc, getDocs, serverTimestamp, setDoc } from 'firebase/firestore';
import * as firestore from 'firebase/firestore';
import { deleteAccountBackup, deleteSetBackup } from '../src/lib/cloudDeletion.js';

let environment;
before(async () => {
  environment = await initializeTestEnvironment({
    projectId: 'demo-signcards-security',
    firestore: { rules: await readFile(new URL('../firestore.rules', import.meta.url), 'utf8') },
  });
});
beforeEach(async () => environment.clearFirestore());
after(async () => environment?.cleanup());

const databaseFor = (uid) => environment.authenticatedContext(uid).firestore();
const baseline = () => ({ refKey: 'fingerspelling:A', h1: 'AAAA', updatedAt: 123, syncedAt: serverTimestamp() });
const customSet = () => ({ cloudId: 'deck-one', title: 'Practice', words: [{ word: 'HELLO', isMultiSign: false, components: [] }], updatedAt: 123, syncedAt: serverTimestamp() });
const history = () => ({ entries: [{ word: 'A', timestamp: 123, status: 'correct' }], syncedAt: serverTimestamp() });

test('owner can write, read, query and delete each supported collection', async () => {
  const database = databaseFor('alice');
  for (const [name, payload] of [['baselines', baseline()], ['customSets', customSet()], ['history', history()]]) {
    const reference = doc(database, `users/alice/${name}/deck-one`);
    await assertSucceeds(setDoc(reference, payload));
    assert.equal((await assertSucceeds(getDoc(reference))).exists(), true);
    assert.equal((await assertSucceeds(getDocs(collection(database, `users/alice/${name}`)))).size, 1);
    await assertSucceeds(deleteDoc(reference));
  }
});

test('signed-out callers and other accounts cannot read, list, overwrite or delete owner data', async () => {
  await setDoc(doc(databaseFor('alice'), 'users/alice/baselines/test'), baseline());
  for (const database of [databaseFor('bob'), environment.unauthenticatedContext().firestore()]) {
    const reference = doc(database, 'users/alice/baselines/test');
    await assertFails(getDoc(reference));
    await assertFails(getDocs(collection(database, 'users/alice/baselines')));
    await assertFails(setDoc(reference, baseline()));
    await assertFails(deleteDoc(reference));
  }
});

test('owner cannot write arbitrary paths, extra fields or client-controlled sync timestamps', async () => {
  const database = databaseFor('alice');
  await assertFails(setDoc(doc(database, 'users/alice/private/anything'), { anything: true }));
  await assertFails(setDoc(doc(database, 'users/alice/baselines/test/nested/test'), baseline()));
  await assertFails(setDoc(doc(database, 'users/alice/baselines/test'), { ...baseline(), unexpected: 'junk' }));
  await assertFails(setDoc(doc(database, 'users/alice/baselines/test'), { ...baseline(), syncedAt: new Date('2100-01-01') }));
});

test('rejects oversized payloads and malformed top-level data', async () => {
  const database = databaseFor('alice');
  await assertFails(setDoc(doc(database, 'users/alice/baselines/test'), { ...baseline(), h1: 'A'.repeat(201604) }));
  await assertFails(setDoc(doc(database, 'users/alice/baselines/test'), { ...baseline(), motion: Array(601).fill({ x: 0 }) }));
  await assertFails(setDoc(doc(database, 'users/alice/customSets/deck-one'), { ...customSet(), words: Array(1001).fill('A') }));
  await assertFails(setDoc(doc(database, 'users/alice/customSets/deck-one'), { ...customSet(), cloudId: 'different-id' }));
  await assertFails(setDoc(doc(database, 'users/alice/history/test'), { ...history(), entries: Array(1001).fill({ word: 'A' }) }));
  await assertFails(setDoc(doc(database, 'users/alice/history/test'), { ...history(), entries: 'invalid' }));
});

test('legacy documents remain readable and deletable by their owner', async () => {
  await environment.withSecurityRulesDisabled(async (context) => {
    await setDoc(doc(context.firestore(), 'users/alice/history/legacy'), { word: 'A', timestamp: 123, status: 'correct' });
  });
  const reference = doc(databaseFor('alice'), 'users/alice/history/legacy');
  await assertSucceeds(getDoc(reference));
  await assertSucceeds(deleteDoc(reference));
});

test('account cleanup removes supported backups, preserves other accounts and prevents stale uploads', async () => {
  const database = databaseFor('alice');
  await setDoc(doc(database, 'users/alice/baselines/test'), baseline());
  await setDoc(doc(database, 'users/alice/customSets/deck-one'), customSet());
  await setDoc(doc(database, 'users/alice/history/test'), history());
  await setDoc(doc(databaseFor('bob'), 'users/bob/baselines/test'), baseline());
  await assertSucceeds(deleteAccountBackup({ ...firestore, db: database }, 'alice'));
  await assertSucceeds(deleteAccountBackup({ ...firestore, db: database }, 'alice'));
  for (const name of ['baselines', 'customSets', 'history']) {
    assert.equal((await getDocs(collection(database, `users/alice/${name}`))).size, 0);
  }
  assert.equal((await getDoc(doc(database, 'users/alice'))).data().deleting, true);
  await assertFails(setDoc(doc(database, 'users/alice/history/stale-tab'), history()));
  await assertFails(setDoc(doc(database, 'users/alice'), { deleting: false, syncedAt: serverTimestamp() }));
  assert.equal((await getDoc(doc(databaseFor('bob'), 'users/bob/baselines/test'))).exists(), true);
  await assertFails(deleteAccountBackup({ ...firestore, db: databaseFor('bob') }, 'alice'));
});

test('set deletion removes its baselines and content but keeps a sync marker that rejects resurrection', async () => {
  const database = databaseFor('alice');
  await setDoc(doc(database, 'users/alice/customSets/deck-one'), customSet());
  const reference = doc(database, 'users/alice/baselines/custom');
  const customBaseline = { ...baseline(), refKey: 'custom:deck-one:HELLO' };
  await setDoc(reference, customBaseline);
  await setDoc(doc(database, 'users/alice/baselines/preset'), baseline());
  await assertSucceeds(deleteSetBackup({ ...firestore, db: database }, 'alice', 'deck-one'));
  assert.equal((await getDoc(reference)).exists(), false);
  assert.equal((await getDoc(doc(database, 'users/alice/baselines/preset'))).exists(), true);
  const deletedSet = (await getDoc(doc(database, 'users/alice/customSets/deck-one'))).data();
  assert.equal(deletedSet.deleted, true);
  assert.equal('words' in deletedSet, false);
  assert.equal('title' in deletedSet, false);
  await assertFails(setDoc(doc(database, 'users/alice/customSets/deck-one'), customSet()));
  await assertFails(setDoc(reference, customBaseline));
});
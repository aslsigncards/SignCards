import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { initializeTestEnvironment, assertFails, assertSucceeds } from '@firebase/rules-unit-testing';
import { deleteObject, getMetadata, listAll, ref, uploadString } from 'firebase/storage';

let environment;
const path = 'users/alice/baselines/fingerspelling_A_h1.b64';
before(async () => {
  environment = await initializeTestEnvironment({
    projectId: 'demo-signcards-security',
    storage: { rules: await readFile(new URL('../storage.rules', import.meta.url), 'utf8') },
  });
  await environment.withSecurityRulesDisabled(async (context) => {
    await uploadString(ref(context.storage(), path), 'AAAA');
  });
});
after(async () => environment?.cleanup());

test('legacy Storage permits owner reads/lists but blocks signed-out and cross-account access', async () => {
  const ownerStorage = environment.authenticatedContext('alice').storage();
  assert.equal((await assertSucceeds(getMetadata(ref(ownerStorage, path)))).size, 4);
  assert.equal((await assertSucceeds(listAll(ref(ownerStorage, 'users/alice/baselines')))).items.length, 1);
  for (const storage of [environment.authenticatedContext('bob').storage(), environment.unauthenticatedContext().storage()]) {
    await assertFails(getMetadata(ref(storage, path)));
    await assertFails(listAll(ref(storage, 'users/alice/baselines')));
    await assertFails(deleteObject(ref(storage, path)));
  }
});

test('even owners cannot upload or overwrite files in legacy Storage', async () => {
  const storage = environment.authenticatedContext('alice').storage();
  await assertFails(uploadString(ref(storage, path), 'replacement'));
  await assertFails(uploadString(ref(storage, 'users/alice/baselines/new.b64'), 'new'));
  await assertFails(uploadString(ref(storage, 'users/alice/arbitrary.txt'), 'junk'));
  await assertSucceeds(deleteObject(ref(storage, path)));
});
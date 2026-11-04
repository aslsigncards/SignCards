async function deleteMatching(fs, collectionRef, constraints = []) {
  while (true) {
    const snapshot = await fs.getDocsFromServer(fs.query(collectionRef, ...constraints, fs.limit(400)));
    if (snapshot.empty) return;
    const batch = fs.writeBatch(fs.db);
    snapshot.docs.forEach((snapshotDoc) => batch.delete(snapshotDoc.ref));
    await batch.commit();
  }
}

export async function deleteAccountBackup(fs, uid) {
  await fs.setDoc(fs.doc(fs.db, `users/${uid}`), { deleting: true, syncedAt: fs.serverTimestamp() });
  for (const name of ['baselines', 'customSets', 'history']) {
    await deleteMatching(fs, fs.collection(fs.db, `users/${uid}/${name}`));
  }
}

export async function deleteSetBackup(fs, uid, cloudId) {
  await fs.setDoc(fs.doc(fs.db, `users/${uid}/customSets/${cloudId}`), {
    cloudId,
    deleted: true,
    updatedAt: Date.now(),
    syncedAt: fs.serverTimestamp(),
  });
  const prefix = `custom:${cloudId}:`;
  await deleteMatching(fs, fs.collection(fs.db, `users/${uid}/baselines`), [
    fs.where('refKey', '>=', prefix),
    fs.where('refKey', '<', `${prefix}\uf8ff`),
  ]);
}
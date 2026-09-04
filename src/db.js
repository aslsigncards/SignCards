import Dexie from 'dexie';

export const db = new Dexie('ASLSignCardsDB');

db.version(1).stores({
  history: '++id, word, timestamp, status',
  customSets: '++id, title, *words'
});

db.version(2).stores({
  history: '++id, word, timestamp, status',
  customSets: '++id, title, *words',
  references: 'word, timestamp'
});

db.version(3)
  .stores({
    history: '++id, word, timestamp, status',
    customSets: '++id, title, *words',
    references: 'word, timestamp'
  })
  .upgrade(async (tx) => {
    const references = tx.table('references');
    const rows = await references.toArray();

    for (const row of rows) {
      if (!row.word || row.word.includes(':')) {
        continue;
      }

      let setName = 'legacy';
      if (/^[A-Z]$/.test(row.word)) {
        setName = 'fingerspelling';
      } else if (/^(10|[1-9])$/.test(row.word)) {
        setName = 'numbers';
      }

      const scopedWord = `${setName}:${row.word}`;
      const alreadyMigrated = await references.get(scopedWord);

      if (!alreadyMigrated) {
        await references.put({ ...row, word: scopedWord });
      }

      await references.delete(row.word);
    }
  });

db.version(4)
  .stores({
    history: '++id, word, timestamp, status',
    references: 'word, timestamp',
    customSets: '++id, title'
  })
  .upgrade(async (tx) => {
    const sets = await tx.table('customSets').toArray();
    for (const set of sets) {
      if (!Array.isArray(set.words) || !set.words.length) continue;
      if (typeof set.words[0] !== 'string') continue;
      const migrated = set.words.map((w) => ({
        word: w,
        isMultiSign: false,
        components: [],
      }));
      await tx.table('customSets').update(set.id, { words: migrated });
    }
  });
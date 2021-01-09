const MAX_REPLAY_MEMORY_SIZE = 500;
const MEMORY_BATCH_SIZE = 64;

class ReplayMemory {
  constructor(size = MAX_REPLAY_MEMORY_SIZE, batchSize = MEMORY_BATCH_SIZE) {
    this.size = size;
    this.batchSize = batchSize;
    this.memory = [];
  }

  addSample(sample) {
    tf.tidy(() => {
      this.memory.push(sample);

      if (this.memory.length > this.size)
        this.memory = this.memory
          //.sort((s1, s2) => s2[2] - s1[2]) // sort by max reward
          .slice(0, this.size)
    });
  }

  getBatch(size) {
    return tf.tidy(() => {
      let shuffled = this.memory.slice(0), i = this.memory.length, temp, index;

      while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
      }
      return shuffled.slice(0, size || this.batchSize);
    })
  }

  dispose() {
    this.memory.map(x => {
      x[0].dispose();
      x[3].dispose();
    });

    this.memory = [];
  }
}

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
        this.memory.shift();
    });
  }

  getBatch(size) {
    return tf.tidy(() => {
      return this.memory.slice(0, size || this.batchSize)
    });
  }

  dispose() {
    tf.tidy(() => {
      this.memory = [];
    });
  }
}

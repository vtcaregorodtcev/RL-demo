class Agent {
  static MOVE_RIGHT = 0;
  static MOVE_DOWN = 1;
  static MOVE_LEFT = 2;
  static MOVE_UP = 3;

  static ACTIONS = [
    Agent.MOVE_RIGHT,
    Agent.MOVE_DOWN,
    Agent.MOVE_LEFT,
    Agent.MOVE_UP
  ];

  constructor(b, r, s, w, h) {
    this.network = b;

    this.rect = r;
    this.speed = s;
    this.width = w;
    this.height = h;
  }

  createModel(inputShape) {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      strides: 1,
      activation: 'relu',
      inputShape,
      dataFormat: 'channelsFirst'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      strides: 1,
      activation: 'relu',
      dataFormat: 'channelsFirst'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      strides: 1,
      activation: 'relu',
      dataFormat: 'channelsFirst'
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.dense({ units: Agent.ACTIONS.length }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return model;
  }

  chooseAction(state, eps) {
    const [_, ...goalShape] = state.shape;

    if (!this.network) this.network = this.createModel(goalShape);

    if (random(0, 1) < eps) {
      return tf.tidy(() => {
        const logits = this.network.predict(state);
        const sigmoid = tf.sigmoid(logits);
        const probs = tf.div(sigmoid, tf.sum(sigmoid));
        return tf.multinomial(probs, 1).dataSync()[0];
      });
    } else {
      return tf.tidy(() => {
        const probs = this.network.predict(state).dataSync();

        return Agent.ACTIONS[probs.indexOf(Math.max(...probs))];
      });
    }
  }

  update(action) {
    switch (action) {
      case Agent.MOVE_UP:
        this.rect.top = this.rect.top - this.speed;
        break;
      case Agent.MOVE_DOWN:
        this.rect.top = this.rect.top + this.speed;
        break;
      case Agent.MOVE_RIGHT:
        this.rect.left = this.rect.left + this.speed;
        break;
      case Agent.MOVE_LEFT:
        this.rect.left = this.rect.left - this.speed;
        break;
    }

    if (this.rect.left < 0) {
      this.rect.left = 0;
    }
    if (this.rect.left > this.width - this.rect.width) {
      this.rect.left = this.width - this.rect.width;
    }
    if (this.rect.top < 0) {
      this.rect.top = 0;
    }
    if (this.rect.top > this.height - this.rect.height) {
      this.rect.top = this.height - this.rect.height;
    }
  }
}

class Environment {
  constructor(w, h, r, c, es, as) {
    this.width = w;
    this.height = h;
    this.rows = r;
    this.columns = c;
    this.enemySpeed = es * this.enemySize;
    this.agentSpeed = as * this.agentSize;

    this.#enemies = this.enemies;

    this.agent = this.resetAgent();
    this.eps = Environment.MAX_EPS;
    this.discount = Environment.DISCOUNT;
  }

  static MAX_EPS = 0.4;
  static MIN_EPS = 0.01;
  static LAMBDA = 0.01;
  static DISCOUNT = 0.99;

  resetAgent() {
    return new Agent(this.agent?.network, {
      left: 0,
      top: 0,
      width: this.agentSize,
      height: this.agentSize,
    }, this.agentSpeed, this.width, this.height)
  };

  reset() {
    this.agent = this.resetAgent();
    // this.#enemies = [];
  };

  get enemyChannel() {
    const channel = [];
    let x = 0, y = 0;

    for (let i = 0; i < this.width; i += this.enemySize) {
      for (let j = 0; j < this.height; j += this.enemySize) {
        if (!channel[x]) channel[x] = [];

        channel[x][y] = this.enemies.some(e => (
          i <= e.left && i + this.enemySize >= (e.right)
        ) && (
            j <= e.top && j + this.enemySize >= (e.bottom)
          )) ? 1 : 0;

        y++;
      }

      x++;
      y = 0;
    }

    return transpose(channel);
  }

  get agentChannel() {
    const channel = [];
    let x = 0, y = 0;

    const { left, right, top, bottom } = toRect(this.agent.rect);

    for (let i = 0; i < this.width; i += this.enemySize) {
      for (let j = 0; j < this.height; j += this.enemySize) {
        if (!channel[x]) channel[x] = [];

        channel[x][y] = (i >= left && i + this.agentSize <= right)
          && (j >= top && j + this.agentSize <= bottom) ? 1 : 0;

        y++;
      }

      x++;
      y = 0;
    }

    return transpose(channel);
  }

  get goalChannel() {
    const channel = [];
    let x = 0, y = 0;

    const { left, right, top, bottom } = toRect(this.goal);

    for (let i = 0; i < this.width; i += this.enemySize) {
      for (let j = 0; j < this.height; j += this.enemySize) {
        if (!channel[x]) channel[x] = [];

        channel[x][y] = (i >= left && i + this.agentSize <= right)
          && (j >= top && j + this.agentSize <= bottom) ? 1 : 0;

        y++;
      }

      x++;
      y = 0;
    }

    return transpose(channel);
  }

  getState() {
    const ec = this.enemyChannel;
    const ac = this.agentChannel;
    const gc = this.goalChannel;

    return [ac, gc, ec];
  }

  getStateTensor() {
    const [ac, gc, ec] = this.getState();

    const buffer = tf.buffer([1, 3, this.rows, this.columns]);

    ac.map((x, ix) => x.map((val, iy) => {
      buffer.set(val, 0, 0, ix, iy);
    }))

    gc.map((x, ix) => x.map((val, iy) => {
      buffer.set(val, 0, 1, ix, iy);
    }))

    ec.map((x, ix) => x.map((val, iy) => {
      buffer.set(val, 0, 2, ix, iy);
    }))

    // Convert the buffer back to a tensor.
    return buffer.toTensor();
  }

  #enemies = [];

  get enemies() {
    if (this.#enemies.length) return this.#enemies;
    else {
      const size = this.enemySize;
      const enemies = [];

      for (let i = 1; i < this.columns - 1; i++) {
        const left = (this.colWidth * i + this.colWidth / 2 - size / 2);
        const top = (randomWithStep(0, this.height - size, this.rowWidth));

        enemies.push({
          left: left + 1, //corrective 1 to not detect wrong intersecting,
          top: top + 1,
          width: size,
          height: size,
          color: '#366',
          speed: this.enemySpeed,
          changableAxis: 'top',
          minValue: 0,
          maxValue: this.height - size,
          get right() {
            return left + size - 1; //corrective 1 to not detect wrong intersecting
          },
          get bottom() {
            return top + size - 1;
          },
        })
      }

      /* for (let i = 1; i < this.rows - 1; i++) {
         const left = (randomWithStep(0, this.width - size, this.colWidth));
         const top = (this.rowWidth * i + this.rowWidth / 2 - size / 2);

         enemies.push({
           left: + 1, //corrective 1 to not detect wrong intersecting,
           top: top + 1,
           width: size,
           height: size,
           right: left + size - 1, //corrective 1 to not detect wrong intersecting
           bottom: top + size - 1,
           color: '#663',
           speed: -this.enemySpeed,
           changableAxis: 'left',
           minValue: 0,
           maxValue: this.height - size
         })
       } */

      return this.#enemies = enemies;
    }
  }

  updateEnemies() {
    this.enemies.map(e => {
      e[e.changableAxis] = e[e.changableAxis] + e.speed;

      if (e[e.changableAxis] < e.minValue || e[e.changableAxis] > e.maxValue) {
        e.speed = -e.speed;
      }
    })
  }

  updateAgent(STATE = this.getStateTensor()) {
    const action = this.agent.chooseAction(STATE, this.eps);

    this.agent.update(action);

    const state = this.getStateTensor();

    return [state, action, this.isDone()];
  }

  calcReward() {
    // penalty less if near goal
    let reward = -distance(
      this.agent.rect.left,
      this.goal.left,
      this.agent.rect.top,
      this.goal.top
    ) / distance(0, this.width, 0, this.height) * 0.2;

    const agentRect = toRect(this.agent.rect);
    const enemiesRects = this.enemies.map(e => toRect(e));
    const goalRect = toRect(this.goal);

    const intersected = enemiesRects.filter(e => rectsIntersected(e, agentRect));
    reward += intersected.length && -10;

    if (rectsIntersected(agentRect, goalRect)) reward += 100;

    return reward;
  }

  isDone() {
    const agentRect = toRect(this.agent.rect);
    const enemiesRects = this.enemies;
    const goalRect = toRect(this.goal);

    return rectsIntersected(agentRect, goalRect) || enemiesRects.some(e => rectsIntersected(e, agentRect));
  }

  decayEps(step) {
    this.eps = Environment.MIN_EPS + (
      Environment.MAX_EPS - Environment.MIN_EPS
    ) * Math.exp(-Environment.LAMBDA * step);
  }

  get goal() {
    return {
      left: this.width - this.colWidth,
      top: this.height - this.rowWidth,
      width: this.colWidth,
      height: this.rowWidth
    }
  }

  get netLineWidth() {
    return 0.1;
  }

  get rowWidth() {
    return this.height / this.rows;
  }

  get colWidth() {
    return this.width / this.columns;
  }

  get agentSize() {
    return this.enemySize;
  }

  get enemySize() {
    const lessInSuchTimes = 1; // 3;

    return min(
      this.rowWidth / lessInSuchTimes,
      this.colWidth / lessInSuchTimes
    );
  }

  get dims() {
    return [this.width, this.height];
  }

  get net() {
    const result = [];
    const rowsLines = this.rows - 1;
    const colsLines = this.columns - 1;

    for (let i = 0; i < rowsLines; i++) {
      result.push({
        left: 0,
        top: this.rowWidth * (i + 1),
        width: this.width,
        height: this.netLineWidth
      })
    }

    for (let i = 0; i < colsLines; i++) {
      result.push({
        left: this.colWidth * (i + 1),
        top: 0,
        width: this.netLineWidth,
        height: this.height
      })
    }

    return result;
  }
}

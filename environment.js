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
    model.add(tf.layers.dense({ inputShape: [inputShape], units: 35, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 35, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 35, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.20 }));
    model.add(tf.layers.dense({ units: 35, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 35, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 35, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 35, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.20 }));
    model.add(tf.layers.dense({ units: Agent.ACTIONS.length }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return model;
  }

  chooseAction(state, eps) {
    if (!this.network) this.network = this.createModel(state.size);

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
        this.rect.top = this.rect.top + this.speed;
        break;
      case Agent.MOVE_DOWN:
        this.rect.top = this.rect.top - this.speed;
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
    this.enemySpeed = es;
    this.agentSpeed = as;

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

  getState() { // normalized
    const maxDistance = distance(0, this.width, 0, this.height);

    const agentX1 = this.agent.rect.left;
    const agentY1 = this.agent.rect.top;

    const agentX2 = agentX1 + this.agentSize;
    const agentY2 = agentY1 + this.agentSize;

    let isThereEnemyInFrontOnX = 0;
    let isThereEnemyInFrontOnY = 0;

    let minDistanceToEnemyOnX = this.width;
    let minDistanceToEnemyOnY = this.height;

    this.enemies.map(e => {
      const eTop2 = e.top + this.enemySize;
      const eLeft2 = e.left + this.enemySize;

      if (e.left > agentX2 && (
        agentY1 <= e.top && e.top <= agentY2
        ||
        agentY1 <= eTop2 && eTop2 <= agentY2
      )) {
        isThereEnemyInFrontOnX = 1;

        minDistanceToEnemyOnX = Math.min(minDistanceToEnemyOnX, distance(agentX2, e.left, agentY2, e.top));
      }

      if (e.top > agentY2 && (
        agentX1 <= e.left && e.left <= agentX2
        ||
        agentX1 <= eLeft2 && eLeft2 <= agentX2
      )) {
        isThereEnemyInFrontOnY = 1;

        minDistanceToEnemyOnY = Math.min(minDistanceToEnemyOnY, distance(agentX2, e.left, agentY2, e.top));
      }
    });

    let isThereGoalInFrontOnX = 0;
    let isThereGoalInFrontOnY = 0;

    let minDistanceToGoalOnX = this.width;
    let minDistanceToGoalOnY = this.height;

    const goalX = this.goal.left;
    const goalY = this.goal.top;

    const goalX2 = goalX + this.colWidth;
    const goalY2 = goalY + this.rowWidth;

    if (goalX > agentX2 && (
      agentY1 <= goalY && goalY <= agentY2
      ||
      agentY1 <= goalY2 && goalY2 <= agentY2
    )) {
      isThereGoalInFrontOnX = 1;

      minDistanceToGoalOnX = Math.min(minDistanceToGoalOnX, distance(agentX2, goalX, agentY2, goalY));
    }

    if (goalY > agentY2 && (
      agentX1 <= goalX && goalX <= agentX2
      ||
      agentX1 <= goalX2 && goalX2 <= agentX2
    )) {
      isThereGoalInFrontOnY = 1;

      minDistanceToGoalOnY = Math.min(minDistanceToGoalOnY, distance(agentX2, goalX, agentY2, goalY));
    }

    const commonGoalDistance = distance(agentX2, goalX, agentY2, goalY);

    return [
      agentX1 / this.width,
      agentX2 / this.width,
      agentY1 / this.height,
      agentY2 / this.height,
      isThereEnemyInFrontOnX,
      isThereEnemyInFrontOnY,
      minDistanceToEnemyOnX / this.width,
      minDistanceToEnemyOnY / this.height,
      isThereGoalInFrontOnX,
      isThereGoalInFrontOnY,
      minDistanceToGoalOnX / this.width,
      minDistanceToGoalOnY / this.height,
      commonGoalDistance / maxDistance
    ];
  }

  getStateTensor() {
    return tf.tensor2d([this.getState()])
  }

  #enemies = [];

  get enemies() {
    if (this.#enemies.length) return this.#enemies;
    else {
      const size = this.enemySize;
      const enemies = [];

      for (let i = 1; i < this.columns - 1; i++) {
        enemies.push({
          left: this.colWidth * i + this.colWidth / 2 - size / 2,
          top: randomWithStep(0, this.height - size, this.rowWidth),
          width: size,
          height: size,
          color: '#366',
          speed: this.enemySpeed,
          changableAxis: 'top',
          minValue: 0,
          maxValue: this.height - size
        })
      }

      for (let i = 1; i < this.rows - 1; i++) {
        enemies.push({
          left: randomWithStep(0, this.width - size, this.colWidth),
          top: this.rowWidth * i + this.rowWidth / 2 - size / 2,
          width: size,
          height: size,
          color: '#663',
          speed: -this.enemySpeed,
          changableAxis: 'left',
          minValue: 0,
          maxValue: this.height - size
        })
      }

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
    const enemiesRects = this.enemies.map(e => toRect(e));
    const goalRect = toRect(this.goal)

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
    const lessInSuchTimes = 3;

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

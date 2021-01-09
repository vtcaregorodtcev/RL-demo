let env = null;
let mem = null;
let STATE = null;

let CURRENT_STEP = 0;
const MAX_STEPS_PER_GAME = 500;

function setup() {
  mem = new ReplayMemory();
  env = new Environment(600, 600, 16, 16, 4, 4);

  [STATE] = env.updateAgent();

  createCanvas(...env.dims);
  goal = loadImage('assets/goal.png');
}

function drawNet(net) {
  net.map(r => {
    const gray = '#ccc';

    fill(gray);
    stroke(gray);
    rect(r.left, r.top, r.width, r.height)
  })
}

function drawEnemies(enemies) {
  enemies.map(e => {
    const black = '#000';

    fill(e.color);
    stroke(black);
    rect(e.left, e.top, e.width, e.height)
  })
}

function drawAgent(r) {
  const green = '#6d6';

  fill(green);
  stroke(green);
  rect(r.left, r.top, r.width, r.height);
}

function drawGoal(r) {
  const padding = 30;

  image(goal, r.left + padding, r.top + padding, r.width - padding * 2, r.height - padding * 2);
}

async function draw() {
  CURRENT_STEP++;

  background(220);

  drawGoal(env.goal);
  drawNet(env.net);
  drawEnemies(env.enemies);
  drawAgent(env.agent.rect);

  env.updateEnemies();

  const reward = env.calcReward();
  const [nextState, action, done] = env.updateAgent();

  mem.addSample([STATE, action, reward, nextState, done]);

  env.decayEps(CURRENT_STEP);

  STATE = nextState;

  if (
    done
    || CURRENT_STEP >= MAX_STEPS_PER_GAME
  ) {
    await replay();

    env.reset();
    CURRENT_STEP = 0;

    mem.dispose();
  }
}

async function replay() {
  const batch = mem.getBatch();

  const states = batch.map(([state, , ,]) => state);
  const nextStates = batch.map(([, , , nextState]) => nextState);

  // Predict the values of each action at each state
  const qsa = states.map((state) => env.agent.network.predict(state));
  // Predict the values of each action at each next state
  const qsad = nextStates.map((nextState) => env.agent.network.predict(nextState));

  let x = new Array();
  let y = new Array();

  // Update the states rewards with the discounted next states rewards
  batch.forEach(
    ([state, action, reward, nextState, done], index) => {

      const currentQ = qsa[index];

      currentQ[action] = !done
        ? reward + env.discount * qsad[index].max().dataSync()
        : reward;

      x.push(state.dataSync());
      y.push(currentQ.dataSync());
    }
  );

  // Clean unused tensors
  qsa.concat(states).forEach((state) => state.dispose());
  qsad.concat(nextStates).forEach((state) => state.dispose());

  // Reshape the batches to be fed to the network
  x = tf.tensor2d(x)
  y = tf.tensor2d(y)

  // Learn the Q(s, a) values given associated discounted rewards
  await env.agent.network.fit(x, y);

  x.dispose();
  y.dispose();
}

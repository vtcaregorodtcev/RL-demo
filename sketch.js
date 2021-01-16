let env = null;
let mem = null;
let STATE = null;

let GOAL_REACHED = false;

let CURRENT_STEP = 0;
const MAX_STEPS_PER_GAME = 1000;

function setup() {
  mem = new ReplayMemory();
  env = new Environment(400, 400, 4, 4, 4, 4);

  [STATE] = env.updateAgent();

  createCanvas(...env.dims);

  cheese = loadImage('assets/cheese.png');
  cat = loadImage('assets/cat.png');
  rat = loadImage('assets/rat.png');
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
    image(cat, e.left, e.top, e.width, e.height);
  })
}

function drawAgent(r) {
  const green = '#6d6';

  fill(green);
  stroke(green);
  image(rat, r.left, r.top, r.width, r.height);
}

function drawGoal(r) {
  const padding = 30;

  image(cheese, r.left + padding, r.top + padding, r.width - padding * 2, r.height - padding * 2);
}

let decayCounter = 1;

async function draw() {
  CURRENT_STEP++;

  background(220);

  drawGoal(env.goal);
  drawNet(env.net);
  drawEnemies(env.enemies);
  drawAgent(env.agent.rect);

  // env.updateEnemies();

  const [nextState, action, done] = env.updateAgent(STATE);

  const reward = env.calcReward();

  mem.append([STATE, action, reward, nextState, done]);

  STATE = nextState;

  if (
    done
    || CURRENT_STEP >= MAX_STEPS_PER_GAME
  ) {
    noLoop();

    await replay();

    await env.agent.network.save(
      `localstorage://jerry-v1-${env.agent.network.inputs[0].shape[1]}`
    );

    env.reset();
    CURRENT_STEP = 0;

    // mem.dispose();

    env.decayEps(decayCounter++);

    loop();
  }
}

async function replay() {
  let miniBatch = mem.sample(500);

  let currentStates = miniBatch.map((dp) => { return dp[0].dataSync() });
  let currentQs = await env.agent.network.predict(tf.tensor(currentStates)).array();
  let newCurrentStates = miniBatch.map((dp) => { return dp[3].dataSync() });
  let futureQs = await env.agent.network.predict(tf.tensor(newCurrentStates)).array();

  let X = [];
  let Y = [];

  for (let index = 0; index < miniBatch.length; index++) {
    const [state, action, reward, newState, done] = miniBatch[index];
    let newQ;
    let currentQ;

    if (!done) {
      let maxFutureQ = Math.max(...futureQs[index]);
      newQ = reward + (env.discount * maxFutureQ);
    }
    else { newQ = reward }

    currentQ = currentQs[index];
    currentQ[action] = newQ;

    X.push(state.dataSync());
    Y.push(currentQ);
  }

  await env.agent.network.fit(tf.tensor(X), tf.tensor(Y), { verbose: 0 });
}

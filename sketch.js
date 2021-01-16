let env = null;
let mem = null;
let STATE = null;

let CURRENT_STEP = 0;
const MAX_STEPS_PER_GAME = 1000;
let GAME_COUNT = 0;
let GOAL_REACHED = 0;

function setup() {
  mem = new ReplayMemory();
  env = new Environment(900, 600, 8, 12, 1, 1);

  [STATE] = env.updateAgent();

  createCanvas(...env.dims);
  frameRate(120);

  playedLabel = createElement('label', `Game played: ${GAME_COUNT} times.`);
  createElement('br');
  reachedLabel = createElement('label', `Goal reached: ${GOAL_REACHED} times`);

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


  if (reward >= 100) GOAL_REACHED++;

  playedLabel.elt.textContent = `Game played: ${GAME_COUNT} times.`
  reachedLabel.elt.textContent = `Goal reached: ${GOAL_REACHED} times.`

  if (
    done
    || CURRENT_STEP >= MAX_STEPS_PER_GAME
  ) {
    noLoop();

    await replay();

    /*await env.agent.network.save(
      `localstorage://jerry-v1-${env.agent.network.inputs[0].shape[1]}`
    );*/

    env.reset();
    CURRENT_STEP = 0;
    GAME_COUNT++;

    // mem.dispose();

    env.decayEps(decayCounter++);

    loop();
  }
}

async function replay() {
  let miniBatch = mem.sample(128);

  const filtered = miniBatch.filter(Boolean);

  if (!filtered.length) return;

  let currentStates = filtered.map((dp) => { return dp[0].dataSync() });
  let currentQs = await env.agent.network.predict(
    tf.tensor(currentStates, [filtered.length, 3, env.rows, env.columns])
  ).array();

  let newCurrentStates = filtered.map((dp) => { return dp[3].dataSync() });
  let futureQs = await env.agent.network.predict(
    tf.tensor(newCurrentStates, [filtered.length, 3, env.rows, env.columns])
  ).array();

  let X = [];
  let Y = [];

  for (let index = 0; index < filtered.length; index++) {
    const [state, action, reward, newState, done] = filtered[index];
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

  await env.agent.network.fit(tf.tensor(X, [filtered.length, 3, env.rows, env.columns]), tf.tensor(Y), { verbose: 0 });
}

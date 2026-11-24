// HANI 10m dash viewer: MuJoCo (WebAssembly) physics + Three.js rendering.
// No neural network yet: the 12 position actuators simply hold the standing pose
// from the "home" keyframe. A trained policy will later write data.ctrl here.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';

const MODEL_URL = 'models/hani_track.xml';
const START_X = 0;      // start line (m)
const FINISH_X = 10;    // finish line (m)
const NOSE = 0.225;     // trunk centre -> front of robot (m)

const GEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6 };

// MuJoCo is z-up, Three.js is y-up: (x, y, z)_mj -> (x, z, -y)_three
const mj2three = (x, y, z) => new THREE.Vector3(x, z, -y);

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- physics
const mj = await loadMujoco();
const xml = await (await fetch(MODEL_URL)).text();
const model = mj.MjModel.from_xml_string(xml);
const data = new mj.MjData(model);
const dt = model.opt.timestep;
const trunkId = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'trunk');
const ACTUATION_OFF = mj.mjtDisableBit.mjDSBL_ACTUATION.value;

function resetSim() {
  mj.mj_resetDataKeyframe(model, data, 0);
  mj.mj_forward(model, data);
  data.xfrc_applied.fill(0);
  setMotors(true);
  pushTimer = 0;
}

// ---------------------------------------------------------------- scene
const container = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = skyTexture();
scene.fog = new THREE.Fog(0xcfe6f5, 30, 80);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 200);
camera.position.copy(mj2three(-1.8, -2.6, 1.1));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(mj2three(-0.35, 0, 0.2));
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 0.6;
controls.maxDistance = 40;

scene.add(new THREE.HemisphereLight(0xdff1ff, 0x5c7a4a, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const sc = sun.shadow.camera;
sc.left = -4; sc.right = 4; sc.top = 4; sc.bottom = -4; sc.near = 0.5; sc.far = 30;
sun.shadow.bias = -0.0005;
scene.add(sun, sun.target);
const SUN_OFFSET = mj2three(-4, -6, 10);

// all MuJoCo geoms live in a z-up group
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);

const meshes = [];       // index = geom id
const dynamic = [];      // geom ids attached to moving bodies
const tmpMat = new THREE.Matrix4();

for (let i = 0; i < model.ngeom; i++) {
  const type = model.geom_type[i];
  const s = model.geom_size.subarray(3 * i, 3 * i + 3);
  const rgba = model.geom_rgba.subarray(4 * i, 4 * i + 4);
  let geo;
  switch (type) {
    case GEOM.PLANE:
      geo = new THREE.PlaneGeometry(2 * (s[0] || 50), 2 * (s[1] || 50));
      break;
    case GEOM.SPHERE:
      geo = new THREE.SphereGeometry(s[0], 24, 16);
      break;
    case GEOM.ELLIPSOID:
      geo = new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
      break;
    case GEOM.CAPSULE:
      geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
      break;
    case GEOM.CYLINDER:
      geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 28).rotateX(Math.PI / 2);
      break;
    case GEOM.BOX:
      geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      break;
    default:
      meshes.push(null);
      continue;
  }
  const name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, i);
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
    roughness: 0.75,
    metalness: 0.05,
    transparent: rgba[3] < 1,
    opacity: rgba[3],
  });
  if (name === 'floor') {
    mat.map = grassTexture();
    mat.roughness = 1;
  } else if (name === 'track') {
    mat.map = trackTexture();
    mat.roughness = 0.95;
  }
  const mesh = new THREE.Mesh(geo, mat);
  const isRobot = model.geom_bodyid[i] !== 0;
  mesh.castShadow = isRobot || type === GEOM.CYLINDER || (type === GEOM.BOX && s[2] > 0.01);
  mesh.receiveShadow = true;
  world.add(mesh);
  meshes.push(mesh);
  if (isRobot) dynamic.push(i);
}

function syncGeom(i) {
  const mesh = meshes[i];
  const p = data.geom_xpos;
  const r = data.geom_xmat;
  const k = 9 * i;
  mesh.position.set(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
  tmpMat.set(
    r[k], r[k + 1], r[k + 2], 0,
    r[k + 3], r[k + 4], r[k + 5], 0,
    r[k + 6], r[k + 7], r[k + 8], 0,
    0, 0, 0, 1,
  );
  mesh.quaternion.setFromRotationMatrix(tmpMat);
}

// ---------------------------------------------------------------- labels
function labelTexture(text, { w = 512, h = 128, fg = '#ffffff', bg = null, font = 'bold 84px system-ui, sans-serif' } = {}) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  if (bg) { g.fillStyle = bg; g.fillRect(0, 0, w, h); }
  g.fillStyle = fg;
  g.font = font;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, w / 2, h / 2 + 4);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}

// text painted on the track, readable when running toward +x
function groundLabel(text, x, y, width, height, opts) {
  const mat = new THREE.MeshStandardMaterial({ map: labelTexture(text, opts), transparent: true, roughness: 0.9, depthWrite: false });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(width, height), mat);
  m.position.copy(mj2three(x, y, 0.0025));
  m.rotation.set(-Math.PI / 2, 0, -Math.PI / 2);
  m.receiveShadow = true;
  scene.add(m);
}

// upright sign facing the direction `face` (MuJoCo coordinates)
function sign(text, pos, face, width, height, opts) {
  const mat = new THREE.MeshStandardMaterial({ map: labelTexture(text, opts), transparent: !opts?.bg, roughness: 0.8 });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(width, height), mat);
  const p = mj2three(...pos);
  m.position.copy(p);
  m.lookAt(p.clone().add(mj2three(...face)));
  scene.add(m);
}

const LANE_W = 1.22;
groundLabel('START', -1.25, 0, 0.55, 2.2, { w: 256, h: 1024, font: 'bold 150px system-ui, sans-serif' });
groundLabel('FINISH', 10.75, 0, 0.55, 2.4, { w: 256, h: 1024, font: 'bold 140px system-ui, sans-serif' });
[1, 2, 3].forEach((lane, k) => {
  groundLabel(String(lane), -2.6, (1 - k) * LANE_W, 0.7, 0.7, { w: 256, h: 256, font: 'bold 200px system-ui, sans-serif' });
});
for (let m = 1; m <= 9; m++) {
  groundLabel(`${m}`, m, -(1.83 + 0.45), 0.35, 0.35, { w: 128, h: 128, font: 'bold 96px system-ui, sans-serif' });
}
sign('FINISH  10 M', [9.93, 0, 2.3], [-1, 0, 0], 4.6, 0.4, { w: 2048, h: 180, bg: '#1c3a8c', font: 'bold 130px system-ui, sans-serif' });
sign('FINISH  10 M', [10.07, 0, 2.3], [1, 0, 0], 4.6, 0.4, { w: 2048, h: 180, bg: '#1c3a8c', font: 'bold 130px system-ui, sans-serif' });
sign('HANI  STADIUM', [5, 1.83 + 5.44, 2.6], [0, -1, 0], 9, 1.0, { w: 2048, h: 228, fg: '#f2f4f8', font: 'bold 150px system-ui, sans-serif' });
['RUN', 'HANI', 'GO!', 'RL', 'RUN'].forEach((t, i) => {
  const opts = { w: 1024, h: 240, fg: '#ffffff', font: 'bold 150px system-ui, sans-serif' };
  sign(t, [-2 + i * 4, -(1.83 + 2.94), 0.45], [0, 1, 0], 3.4, 0.8, opts);
  sign(t, [-2 + i * 4, -(1.83 + 3.06), 0.45], [0, -1, 0], 3.4, 0.8, opts);
});

// ---------------------------------------------------------------- procedural textures
function skyTexture() {
  const c = document.createElement('canvas');
  c.width = 16; c.height = 512;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, '#5aa7e6');
  grad.addColorStop(0.55, '#a9d4f2');
  grad.addColorStop(1, '#e6f3fb');
  g.fillStyle = grad;
  g.fillRect(0, 0, 16, 512);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function noiseCanvas(size, base, jitter, stripes) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const n = (Math.random() - 0.5) * jitter + (stripes && Math.floor(x / (size / 8)) % 2 ? 14 : 0);
      const o = 4 * (y * size + x);
      img.data[o] = base + n; img.data[o + 1] = base + n; img.data[o + 2] = base + n; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return c;
}

function grassTexture() {
  const t = new THREE.CanvasTexture(noiseCanvas(256, 225, 40, true));
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(10, 10);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function trackTexture() {
  const t = new THREE.CanvasTexture(noiseCanvas(256, 230, 50, false));
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(36, 5);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ---------------------------------------------------------------- UI
let paused = false;
let motorsOn = true;
const CAM_MODES = ['추적', '전체', '자유'];
let camMode = 0;
let pushTimer = 0;

function setMotors(on) {
  motorsOn = on;
  model.opt.disableflags = on ? model.opt.disableflags & ~ACTUATION_OFF : model.opt.disableflags | ACTUATION_OFF;
  const b = $('btn-motor');
  b.textContent = on ? '모터 끄기' : '모터 켜기';
  b.classList.toggle('off', !on);
  b.insertAdjacentHTML('beforeend', '<kbd>M</kbd>');
}

function push() {
  // short sideways shove on the trunk (~0.1 s)
  const f = data.xfrc_applied;
  const dir = Math.random() < 0.5 ? -1 : 1;
  f[6 * trunkId + 0] = (Math.random() - 0.5) * 40;
  f[6 * trunkId + 1] = dir * 90;
  pushTimer = 0.1;
}

function togglePause() {
  paused = !paused;
  $('btn-pause').innerHTML = (paused ? '재생' : '일시정지') + '<kbd>Space</kbd>';
}

function toggleCam() {
  camMode = (camMode + 1) % CAM_MODES.length;
  $('btn-cam').innerHTML = `카메라: ${CAM_MODES[camMode]}<kbd>C</kbd>`;
  if (CAM_MODES[camMode] === '전체') {
    // whole 10 m track seen from the billboard side, stadium behind
    camera.position.copy(mj2three(4.5, -11.5, 5.5));
    controls.target.copy(mj2three(4.5, 0, 0));
  } else if (CAM_MODES[camMode] === '추적') {
    camera.position.copy(trunkPos).add(mj2three(-1.45, -2.6, 0.9));
    controls.target.copy(trunkPos);
  }
}

$('btn-reset').onclick = resetSim;
$('btn-motor').onclick = () => setMotors(!motorsOn);
$('btn-push').onclick = push;
$('btn-pause').onclick = togglePause;
$('btn-cam').onclick = toggleCam;
window.addEventListener('keydown', (e) => {
  if (e.repeat) return;
  const k = e.key.toLowerCase();
  if (k === 'r') resetSim();
  else if (k === 'm') setMotors(!motorsOn);
  else if (k === 'p') push();
  else if (k === 'c') toggleCam();
  else if (e.code === 'Space') { e.preventDefault(); togglePause(); }
});
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------- loop
resetSim();
for (let i = 0; i < model.ngeom; i++) if (meshes[i]) syncGeom(i);
$('loading').classList.add('hidden');

let last = performance.now();
let acc = 0;
const trunkPos = new THREE.Vector3();
const prevTarget = new THREE.Vector3();

function frame(now) {
  const elapsed = Math.min((now - last) / 1000, 0.05);
  last = now;

  if (!paused) {
    acc += elapsed;
    while (acc >= dt) {
      mj.mj_step(model, data);
      if (pushTimer > 0) {
        pushTimer -= dt;
        if (pushTimer <= 0) data.xfrc_applied.fill(0);
      }
      acc -= dt;
    }
  }

  for (const i of dynamic) syncGeom(i);

  const q = data.qpos;
  const qv = data.qvel;
  trunkPos.copy(mj2three(q[0], q[1], q[2]));

  if (CAM_MODES[camMode] === '추적') {
    prevTarget.copy(controls.target);
    controls.target.lerp(trunkPos, 0.08);
    camera.position.add(controls.target.clone().sub(prevTarget));
  }
  controls.update();

  sun.target.position.copy(trunkPos);
  sun.position.copy(trunkPos).add(SUN_OFFSET);

  const front = q[0] + NOSE;
  const dist = Math.max(0, front - START_X);
  const fallen = q[2] < 0.12;
  $('time').textContent = `${data.time.toFixed(2)} s`;
  $('dist').textContent = `${Math.min(dist, FINISH_X).toFixed(2)} m`;
  $('speed').textContent = `${qv[0].toFixed(2)} m/s`;
  $('height').textContent = `${q[2].toFixed(2)} m`;
  $('status').textContent = paused ? '일시정지'
    : front >= FINISH_X ? '골인!'
    : fallen ? '넘어짐'
    : motorsOn ? '서 있음 (신경망 없음)'
    : '모터 꺼짐';

  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

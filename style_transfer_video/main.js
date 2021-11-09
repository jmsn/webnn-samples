'use strict';

import {FastStyleTransferNet} from './fast_style_transfer_net.js';
import {showProgressComponent, readyShowResultComponents} from '../common/ui.js';
import * as utils from '../common/utils.js';

const maxWidth = 380;
const maxHeight = 380;
let modelId = 'starry-night';
let isFirstTimeLoad = true;
let isModelChanged = false;
let frameReq;
let fastStyleTransferNet;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let outputBuffer;
let devicePreference = 'gpu';
let lastDevicePreference = '';
let player;
let enabled = true;

$(document).ready(() => {
  $('.icdisplay').hide();
  $('.badge').html(modelId);
});

$('#deviceBtns .btn').on('change', async (e) => {
  devicePreference = $(e.target).attr('id');

  if (player) {
    player.getVideoElement().cancelVideoFrameCallback(frameReq);
  }

  await main();
});

$('#onoffBtns .btn').on('change', async (e) => {
  enabled = $(e.target).attr('id') === 'on';

  if (player) {
    renderVideoFrame(player.getVideoElement());
  }
});

$('#gallery .gallery-image').hover((e) => {
  const id = $(e.target).attr('id');
  const modelName = $('#' + id).attr('title');
  $('.badge').html(modelName);
}, () => {
  const modelName = $(`#${modelId}`).attr('title');
  $('.badge').html(modelName);
});

$('#gallery .gallery-item').click(async (e) => {
  const newModelId = $(e.target).attr('id');

  if (player) {
    player.getVideoElement().cancelVideoFrameCallback(frameReq);
  }

  if (newModelId !== modelId) {
    isModelChanged = true;
    modelId = newModelId;
    const modelName = $(`#${modelId}`).attr('title');
    $('.badge').html(modelName);
    $('#gallery .gallery-item').removeClass('hl');
    $(e.target).parent().addClass('hl');
  }

  await main();
});

async function renderVideoFrame(videoElement) {
  if (!enabled) {
    $('#fps').hide();
    drawFromImageSource(videoElement, 'outputCanvas');
    frameReq = videoElement.requestVideoFrameCallback(function() {
      renderVideoFrame(videoElement);
    });
  } else {
    const inputBuffer =
          utils.getInputTensor(videoElement, fastStyleTransferNet.inputOptions);
    console.log('- Computing... ');
    const start = performance.now();
    fastStyleTransferNet.compute(inputBuffer, outputBuffer);
    computeTime = (performance.now() - start).toFixed(2);
    console.log(`  done in ${computeTime} ms.`);
    videoElement.width = videoElement.videoWidth;
    videoElement.height = videoElement.videoHeight;
    drawFromImageSource(videoElement, 'inputCanvas');
    showPerfResult();
    const inputCanvas = document.getElementById('inputCanvas');
    drawImageData(
        bufferToImageData(outputBuffer),
        'outputCanvas',
        inputCanvas.width,
        inputCanvas.height,
    );
    $('#fps').show();
    $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
    frameReq = videoElement.requestVideoFrameCallback(function() {
      renderVideoFrame(videoElement);
    });
  }
}

function drawFromImageSource(srcElement, canvasId) {
  const canvas = document.getElementById(canvasId);
  const resizeRatio = Math.max(
      Math.max(srcElement.width / maxWidth, srcElement.height / maxHeight), 1);
  const scaledWidth = Math.floor(srcElement.width / resizeRatio);
  const scaledHeight = Math.floor(srcElement.height / resizeRatio);
  canvas.height = scaledHeight;
  canvas.width = scaledWidth;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(srcElement, 0, 0, scaledWidth, scaledHeight);
}

function bufferToImageData(buffer) {
  const outputSize = fastStyleTransferNet.outputDimensions;
  const height = outputSize[2];
  const width = outputSize[3];
  const mean = [1, 1, 1, 1];
  const offset = [0, 0, 0, 0];
  const bytes = new Uint8ClampedArray(width * height * 4);
  const a = 255;

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const r = buffer[i] * mean[0] + offset[0];
    const g = buffer[i + height * width] * mean[1] + offset[1];
    const b = buffer[i + height * width * 2] * mean[2] + offset[2];
    bytes[j + 0] = Math.round(r);
    bytes[j + 1] = Math.round(g);
    bytes[j + 2] = Math.round(b);
    bytes[j + 3] = Math.round(a);
  }

  return {
    data: new ImageData(bytes, width, height),
    height,
    width,
  };
}

function drawImageData(img, outCanvasId, canvasWidth, canvasHeight) {
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = img.width;
  tmpCanvas.height = img.height;

  const outCtx = tmpCanvas.getContext('2d');
  outCtx.putImageData(
      img.data,
      0,
      0,
      0,
      0,
      tmpCanvas.width,
      tmpCanvas.height,
  );

  const outputCanvas = document.getElementById(outCanvasId);
  outputCanvas.width = canvasWidth;
  outputCanvas.height = canvasHeight;

  const ctx = outputCanvas.getContext('2d');
  ctx.drawImage(tmpCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
}

function showPerfResult(medianComputeTime = undefined) {
  $('#loadTime').html(`${loadTime} ms`);
  $('#buildTime').html(`${buildTime} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime} ms`);
  }
}

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

export async function main() {
  try {
    let start;
    // Set 'numRuns' param to run inference multiple times
    const params = new URLSearchParams(location.search);
    let numRuns = params.get('numRuns');
    numRuns = numRuns === null ? 1 : parseInt(numRuns);

    if (numRuns < 1) {
      addWarning('The value of param numRuns must be greater than or equal' +
          ' to 1.');
      return;
    }
    // Only do load() and build() when model first time loads,
    // there's new model choosed, and device backend changed
    if (isFirstTimeLoad || isModelChanged ||
        lastDevicePreference != devicePreference) {
      if (lastDevicePreference != devicePreference) {
        // Set polyfill backend
        await utils.setPolyfillBackend(devicePreference);
        lastDevicePreference = devicePreference;
      }
      if (fastStyleTransferNet !== undefined) {
        // Call dispose() to and avoid memory leak
        fastStyleTransferNet.dispose();
      }
      fastStyleTransferNet = new FastStyleTransferNet();
      outputBuffer = new Float32Array(
          utils.sizeOfShape(fastStyleTransferNet.outputDimensions));
      isFirstTimeLoad = false;
      isModelChanged = false;
      console.log(`- Model ID: ${modelId} -`);
      // UI shows model loading progress
      await showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      start = performance.now();
      const outputOperand =
          await fastStyleTransferNet.load(devicePreference, modelId);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      fastStyleTransferNet.build(outputOperand);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await showProgressComponent('done', 'done', 'current');

    if (!player) {
      const conf = {
        key: 'yolo',
        playback: {
          // autoplay: true,
          muted: true,
        },
        location: {
          ui: 'https://cdn.bitmovin.com/player/web/8/bitmovinplayer-ui.js',
          ui_css: 'https://cdn.bitmovin.com/player/web/8/bitmovinplayer-ui.css',
        },
        adaptation: {
          preload: false,
        },
      };

      // eslint-disable-next-line max-len
      player = new window.bitmovin.player.Player(document.getElementById('player'), conf);

      const source = {
        // eslint-disable-next-line max-len
        dash: '//bitmovin-a.akamaihd.net/content/MI201109210084_1/mpds/f08e80da-bf1d-4e3d-8899-f0f6155f6efa.mpd',
      };

      player.load(source).then(function() {
        const qualities = player.getAvailableVideoQualities();
        player.setVideoQuality(qualities[0].id);
        player.preload();

        const outputCanvas = document.createElement('canvas');
        outputCanvas.id = 'outputCanvas';
        const playerDiv = document.querySelector('#player');
        playerDiv.insertBefore(outputCanvas, playerDiv.firstChild);
        player.getVideoElement().hidden = true;

        renderVideoFrame(player.getVideoElement());
      });

      window.player = player;
    } else {
      renderVideoFrame(player.getVideoElement());
    }

    await showProgressComponent('done', 'done', 'done');
    readyShowResultComponents();
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}

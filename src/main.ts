import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import FFT from "fft.js";

// Define types for working with audio
declare global {
  interface Window {
    webkitAudioContext: typeof AudioContext;
  }
}

const sampleRate = 16000; // Assuming a sample rate of 16kHz
const fftSize = 512; // The size of the FFT
const hopLength = fftSize / 4; // The hop length for overlap (25% overlap)
const nMelBins = 128; // Number of Mel bins for spectrogram

let model: tf.LayersModel | null = null;

// Load the model
(async function loadModel() {
  try {
    model = await tf.loadLayersModel("http://172.19.184.131:8080/model.json");
    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading the model:", error);
  }
})();

// Convert audio to spectrogram
async function audioToSpectrogram(
  audioBuffer: ArrayBuffer
): Promise<Float32Array[]> {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const decodedAudio = await audioContext.decodeAudioData(audioBuffer);

  const numFrames = Math.floor((decodedAudio.length - fftSize) / hopLength) + 1;
  const spectrogram = new Array(numFrames)
    .fill(0)
    .map(() => new Float32Array(nMelBins));

  const audioData = decodedAudio.getChannelData(0);

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopLength;
    const frame = audioData.slice(start, start + fftSize);

    const fft = applyFFT(frame);

    const melSpectrogram = fftToMelSpectrogram(fft);
    spectrogram[i] = new Float32Array(melSpectrogram); // Convert to Float32Array
  }

  return spectrogram;
}

// Event listener for button click
document
  .getElementById("denoise-button")!
  .addEventListener("click", async () => {
    const fileInput = document.getElementById(
      "audio-upload"
    ) as HTMLInputElement;
    if (fileInput.files!.length === 0) {
      alert("Please upload an audio file");
      return;
    }

    const audioFile = fileInput.files![0];
    const audioBuffer = await readAudioFile(audioFile);

    const spectrogram = await audioToSpectrogram(audioBuffer);

    // Flatten the 2D spectrogram array (Float32Array[]) into a 1D array
    const flattenedSpectrogram = spectrogram.reduce(
      (acc, curr) => [...acc, ...curr],
      [] as number[]
    );

    if (model) {
      const spectrogramTensor = tf.tensor4d(flattenedSpectrogram, [
        1,
        spectrogram.length,
        nMelBins,
        1,
      ]);
      const prediction = model.predict(spectrogramTensor) as tf.Tensor;

      const denoisedAudioBuffer = await spectrogramToAudio(prediction);

      playAudioBuffer(denoisedAudioBuffer);
    } else {
      console.error("Model is not loaded yet");
    }
  });

// Function to read the audio file
async function readAudioFile(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      resolve(event.target?.result as ArrayBuffer);
    };
    reader.onerror = (error) => reject(error);
    reader.readAsArrayBuffer(file);
  });
}

// Convert predicted spectrogram back to audio
async function spectrogramToAudio(
  predictedSpectrogram: tf.Tensor
): Promise<ArrayBuffer> {
  const predictedSpectrogramArray =
    predictedSpectrogram.arraySync() as number[][];

  const numFrames = predictedSpectrogramArray.length;
  const reconstructedAudio = new Float32Array(numFrames * hopLength + fftSize);

  for (let i = 0; i < numFrames; i++) {
    const melSpectrogram = predictedSpectrogramArray[i];
    const frame = melSpectrogramToAudioFrame(new Float32Array(melSpectrogram)); // Convert to Float32Array

    for (let j = 0; j < frame.length; j++) {
      reconstructedAudio[i * hopLength + j] += frame[j];
    }
  }

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = audioContext.createBuffer(
    1,
    reconstructedAudio.length,
    sampleRate
  );
  audioBuffer.copyToChannel(reconstructedAudio, 0, 0);

  return bufferToWav(audioBuffer);
}

function applyFFT(frame: Float32Array): Float32Array {
  const f = new FFT(fftSize); // Create an FFT instance
  const out = f.createComplexArray(); // Create a complex array for output
  const input = Array.from(frame); // Convert Float32Array to a normal array for FFT

  // Apply FFT
  f.realTransform(out, input);
  f.completeSpectrum(out); // Get the full FFT spectrum

  // Only return the magnitude (real part)
  const magnitudes = new Float32Array(fftSize / 2 + 1);
  for (let i = 0; i < magnitudes.length; i++) {
    magnitudes[i] = Math.sqrt(out[2 * i] ** 2 + out[2 * i + 1] ** 2); // Magnitude from real/imaginary parts
  }

  return magnitudes;
}

// Mel filter bank generation
function generateMelFilterBank(
  sampleRate: number,
  numMelBins: number,
  fftSize: number
): Float32Array[] {
  const melFilters: Float32Array[] = [];
  const melMin = 0;
  const melMax = 2595 * Math.log10(1 + sampleRate / 2 / 700); // Convert frequency to Mel scale
  const melPoints = new Float32Array(numMelBins + 2);

  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] =
      700 *
      (Math.pow(
        10,
        (melMin + (i / (numMelBins + 1)) * (melMax - melMin)) / 2595
      ) -
        1);
  }

  const bin = melPoints.map((melPoint) =>
    Math.floor(((fftSize + 1) * melPoint) / sampleRate)
  );

  for (let m = 1; m < melPoints.length - 1; m++) {
    const filterBank = new Float32Array(fftSize / 2 + 1);
    for (let k = bin[m - 1]; k < bin[m]; k++) {
      filterBank[k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1]);
    }
    for (let k = bin[m]; k < bin[m + 1]; k++) {
      filterBank[k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m]);
    }
    melFilters.push(filterBank);
  }

  return melFilters;
}

// Apply Mel filter bank to FFT result
function fftToMelSpectrogram(fft: Float32Array): Float32Array {
  const melFilterBank = generateMelFilterBank(sampleRate, nMelBins, fftSize);
  const melSpectrogram = new Float32Array(nMelBins);

  for (let i = 0; i < nMelBins; i++) {
    let sum = 0;
    for (let j = 0; j < fft.length; j++) {
      sum += fft[j] * melFilterBank[i][j];
    }
    melSpectrogram[i] = Math.log10(sum + 1e-6); // Log-scaled mel spectrogram
  }

  return melSpectrogram;
}

// Inverse Mel Filter Bank
function melToLinearSpectrogram(melSpectrogram: Float32Array): Float32Array {
  const melFilterBank = generateMelFilterBank(sampleRate, nMelBins, fftSize);

  const linearSpectrogram = new Float32Array(fftSize / 2 + 1);
  for (let i = 0; i < melSpectrogram.length; i++) {
    for (let j = 0; j < linearSpectrogram.length; j++) {
      linearSpectrogram[j] += melSpectrogram[i] * melFilterBank[i][j];
    }
  }

  return linearSpectrogram;
}

// Inverse FFT using fft.js
function inverseFFT(linearSpectrogram: Float32Array): Float32Array {
  const f = new FFT(fftSize);
  const complexArray = f.createComplexArray(); // Create a complex array for the FFT input

  // Convert linear spectrogram into real and imaginary parts for iFFT
  for (let i = 0; i < linearSpectrogram.length; i++) {
    complexArray[2 * i] = linearSpectrogram[i]; // Real part
    complexArray[2 * i + 1] = 0; // Imaginary part is 0 because we don't have phase info
  }

  const timeDomainFrame = new Float32Array(fftSize);
  f.inverseTransform(timeDomainFrame, complexArray); // Apply inverse FFT

  return timeDomainFrame;
}

// Inverse Mel Spectrogram to Audio Frame
function melSpectrogramToAudioFrame(
  melSpectrogram: Float32Array
): Float32Array {
  // Step 1: Convert Mel spectrogram back to linear spectrogram
  const linearSpectrogram = melToLinearSpectrogram(melSpectrogram);

  // Step 2: Apply inverse FFT to convert linear spectrogram back to time-domain audio frame
  const audioFrame = inverseFFT(linearSpectrogram);

  return audioFrame;
}

// Convert AudioBuffer to WAV
function bufferToWav(audioBuffer: AudioBuffer): ArrayBuffer {
  const numOfChan = audioBuffer.numberOfChannels;
  const length = audioBuffer.length * numOfChan * 2 + 44;
  const buffer = new ArrayBuffer(length);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + audioBuffer.length * numOfChan * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numOfChan, true);
  view.setUint32(24, audioBuffer.sampleRate, true);
  view.setUint32(28, audioBuffer.sampleRate * numOfChan * 2, true);
  view.setUint16(32, numOfChan * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, audioBuffer.length * numOfChan * 2, true);

  let offset = 44;
  for (let i = 0; i < audioBuffer.length; i++) {
    for (let channel = 0; channel < numOfChan; channel++) {
      const sample = audioBuffer.getChannelData(channel)[i] * 32768;
      view.setInt16(offset, sample < 0 ? sample : sample, true);
      offset += 2;
    }
  }

  return buffer;
}

// Write a string to DataView
function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// Play AudioBuffer
function playAudioBuffer(audioBuffer: ArrayBuffer) {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  audioContext.decodeAudioData(audioBuffer, (buffer) => {
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start(0);

    const blob = new Blob([audioBuffer], { type: "audio/wav" });
    const audioURL = URL.createObjectURL(blob);
    const audioElement = document.getElementById(
      "output-audio"
    ) as HTMLAudioElement;
    audioElement.src = audioURL;
  });
}

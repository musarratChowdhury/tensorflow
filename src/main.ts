import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

let modelPromise;

window.onload = async () => {
  const img = document.getElementById("img") as HTMLImageElement;
  modelPromise = cocoSsd.load();
  const model = await modelPromise;
  console.log(model);
  const predictions = await model.detect(img);
  console.log(predictions);
};

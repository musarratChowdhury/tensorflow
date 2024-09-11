import * as tf from "@tensorflow/tfjs";
const dataArray = [8, 6, 7, 5, 3, 0, 9];
const first = tf.tensor(dataArray);
console.log(first);
const first_again = tf.tensor1d(dataArray);
// const guess = tf.tensor2d([true, false, false], undefined, "int32");
const d = tf.tensor([1, 0, 0, 0, -1, 0, 1, 0, 0], [3, 3], "int32");
console.log(d);

import * as tf from "@tensorflow/tfjs";
import "babel-polyfill";

const inputData = tf.tensor2d([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]);
const responseData = tf.tensor1d([0, 1, 1, 0]);

const model = tf.sequential();

model.add(tf.layers.dense({ units: 10, activation: "relu", inputShape: [2] }));
model.add(tf.layers.dense({ units: 1, inputDIm: 1 }));


model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

model
  .fit(inputData, responseData, { verbose: 0 , shuffle: true,  epochs: 2000 })
  .then((res, rej) => {
    
    
    // model
    //   .predict(tf.tensor2d([[0, 0]]))
    //   .reshape([1])
    //   .asScalar()
    //   .print();

    model.predict(tf.tensor2d([[1, 1]])).print()
  });

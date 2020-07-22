//initial variables
let model,
    video, 
    keypoints, 
    brain_network,
    train_data = [],
    train_state = "not_yet_trained",
    saveState = 'waiting',
    poseLabel = "idle",//if pose label == 0 then activity = idle or if pose label == 1 then activity = keep
    predictions=[];

    var unityInstance = UnityLoader.instantiate("unityContainer", "../Build/Desktop.json", {onProgress: UnityProgress});

//listening key pressed
function keyPressed(){
    if(key == 's'){
        //brain.saveData();
        console.log("Saving Training Data...");
        trainNetwork();
    }
    else{
        if(key == 'q'){
            poseLabel = "idle";
            console.log("idle data");
        }
        else if(key == 'w'){
            poseLabel = "keep";
            console.log("keep data");
        }
        setTimeout(function(){
            console.log('collecting train data');
            saveState = 'collecting';
            setTimeout(function(){
                console.log('collecting complete');
                saveState = 'waiting';
            },10000);
        },5000);
    }
}

//hand pose codes start ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
function preload() {
    video = createCapture(VIDEO, () => {
      loadHandTrackingModel();
    });
    video.hide();
}

async function loadHandTrackingModel() {
    // Load the MediaPipe handpose model.
    model = await handpose.load();
    predictHand();
}

async function predictHand() {
    // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain a
    // hand prediction from the MediaPipe graph.
    predictions = await model.estimateHands(video.elt);
    //console.log('predictions: ', predictions)
  
    setTimeout(() => predictHand(), 50);
}

function draw() {
    background(255);
    if (model) image(video, 0, 0);
    if (predictions.length > 0) {
      // We can call both functions to draw all keypoints and the skeletons
      drawKeypoints();
      //drawSkeleton();
      //console.log(predictions[0].landmarks);
      if(saveState == 'collecting' && predictions[0].landmarks.length==21){
        let prediction = predictions[0].landmarks;

        let dist1 = dist(prediction[20][0],prediction[20][1],prediction[0][0],prediction[1][1]);
        let dist2 = dist(prediction[16][0],prediction[16][1],prediction[0][0],prediction[1][1]);
        let dist3 = dist(prediction[12][0],prediction[12][1],prediction[0][0],prediction[1][1]);
        let dist4 = dist(prediction[8][0],prediction[8][1],prediction[0][0],prediction[1][1]);
        let dist5 = dist(prediction[4][0],prediction[4][1],prediction[0][0],prediction[1][1]);

        if(poseLabel=="idle"){
          data = {input: [dist1,dist2,dist3,dist4,dist5], output: { idle : 1}};
        }
        else if(poseLabel == "keep"){
          data = {input: [dist1,dist2,dist3,dist4,dist5], output: { keep : 1}};
        }
        train_data.push(data);
      }
      if(train_state =="trained" && predictions[0].landmarks.length==21){
        let prediction = predictions[0].landmarks;

        let dist1 = dist(prediction[20][0],prediction[20][1],prediction[0][0],prediction[1][1]);
        let dist2 = dist(prediction[16][0],prediction[16][1],prediction[0][0],prediction[1][1]);
        let dist3 = dist(prediction[12][0],prediction[12][1],prediction[0][0],prediction[1][1]);
        let dist4 = dist(prediction[8][0],prediction[8][1],prediction[0][0],prediction[1][1]);
        let dist5 = dist(prediction[4][0],prediction[4][1],prediction[0][0],prediction[1][1]);

        let result =brain_network.run([dist1,dist2,dist3,dist4,dist5]);
        //console.log(result['idle']);
        let estimation =""
        if(result['keep']>result['idle']){
          estimation = "keep"+"X"+Math.trunc((prediction[14][0]+prediction[1][0])/2).toString()+"X"+Math.trunc((prediction[14][1]+prediction[1][1])/2).toString();
        }
        else{
          estimation = "idle"+"X"+Math.trunc((prediction[14][0]+prediction[1][0])/2).toString()+"X"+Math.trunc((prediction[14][1]+prediction[1][1])/2).toString();
        }
        console.log(estimation);
        unityInstance.SendMessage('GameController','changeHandPosition',estimation);
      }
    }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints()  {
    let prediction = predictions[0];
    for (let j = 0; j < prediction.landmarks.length; j++) {
      let keypoint = prediction.landmarks[j];
      fill(j*10, 0, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
}

// A function to draw the skeletons
function drawSkeleton() {
    let annotations = predictions[0].annotations;
    stroke(255, 0, 0);
    for (let j = 0; j < annotations.thumb.length - 1; j++) {
      line(annotations.thumb[j][0], annotations.thumb[j][1], annotations.thumb[j + 1][0], annotations.thumb[j + 1][1]);
    }
    for (let j = 0; j < annotations.indexFinger.length - 1; j++) {
      line(annotations.indexFinger[j][0], annotations.indexFinger[j][1], annotations.indexFinger[j + 1][0], annotations.indexFinger[j + 1][1]);
    }
    for (let j = 0; j < annotations.middleFinger.length - 1; j++) {
      line(annotations.middleFinger[j][0], annotations.middleFinger[j][1], annotations.middleFinger[j + 1][0], annotations.middleFinger[j + 1][1]);
    }
    for (let j = 0; j < annotations.ringFinger.length - 1; j++) {
      line(annotations.ringFinger[j][0], annotations.ringFinger[j][1], annotations.ringFinger[j + 1][0], annotations.ringFinger[j + 1][1]);
    }
    for (let j = 0; j < annotations.pinky.length - 1; j++) {
      line(annotations.pinky[j][0], annotations.pinky[j][1], annotations.pinky[j + 1][0], annotations.pinky[j + 1][1]);
    }

    line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.thumb[0][0], annotations.thumb[0][1]);
    line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.indexFinger[0][0], annotations.indexFinger[0][1]);
    line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.middleFinger[0][0], annotations.middleFinger[0][1]);
    line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.ringFinger[0][0], annotations.ringFinger[0][1]);
    line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.pinky[0][0], annotations.pinky[0][1]);

}
//hand pose codes end ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//deepth learning codes start ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
function setup(){
  createCanvas(1000, 1000);
  const config = {
    inputSize: 5,
    outpuSize: 2,
    hiddenLayers: [3],
    activation: 'sigmoid',
    learningRate: 0.01,
    decayRate: 0.999
  };
  brain_network = new brain.NeuralNetwork(config);
}

function trainNetwork(){
  brain_network.train(train_data);
  console.log("Training Model.");
  train_state = "trained";
}


const itemMap = {
  0: 'person',
  1: 'bicycle',
  2: 'car',
  3: 'motorcycle',
  4: 'airplane',
  5: 'bus',
  6: 'train',
  7: 'truck',
  8: 'boat',
  9: 'traffic light',
  10: 'fire hydrant',
  11: 'stop sign',
  12: 'parking meter',
  13: 'bench',
  14: 'bird',
  15: 'cat',
  16: 'dog',
  17: 'horse',
  18: 'sheep',
  19: 'cow',
  20: 'elephant',
  21: 'bear',
  22: 'zebra',
  23: 'giraffe',
  24: 'backpack',
  25: 'umbrella',
  26: 'handbag',
  27: 'tie',
  28: 'suitcase',
  29: 'frisbee',
  30: 'skis',
  31: 'snowboard',
  32: 'sports ball',
  33: 'kite',
  34: 'baseball bat',
  35: 'baseball glove',
  36: 'skateboard',
  37: 'surfboard',
  38: 'tennis racket',
  39: 'bottle',
  40: 'wine glass',
  41: 'cup',
  42: 'fork',
  43: 'knife',
  44: 'spoon',
  45: 'bowl',
  46: 'banana',
  47: 'apple',
  48: 'sandwich',
  49: 'orange',
  50: 'broccoli',
  51: 'carrot',
  52: 'hot dog',
  53: 'pizza',
  54: 'donut',
  55: 'cake',
  56: 'chair',
  57: 'couch',
  58: 'potted plant',
  59: 'bed',
  60: 'dining table',
  61: 'toilet',
  62: 'tv',
  63: 'laptop',
  64: 'mouse',
  65: 'remote',
  66: 'keyboard',
  67: 'cell phone',
  68: 'microwave',
  69: 'oven',
  70: 'toaster',
  71: 'sink',
  72: 'refrigerator',
  73: 'book',
  74: 'clock',
  75: 'vase',
  76: 'scissors',
  77: 'teddy bear',
  78: 'hair drier',
  79: 'toothbrush'
};

const colorMap = {
  0: 'red',
  1: 'blue',
  2: 'green',
  3: 'cyan',
  4: 'magenta',
  5: 'beige',
  6: 'purple',
  7: 'yellow',
  8: 'pink',
  9: 'orange'

}

function flipText(text) {
  // Split the text into an array of lines
  const lines = text.split('\n');

  // Transpose the lines (flip 90 degrees to the left)
  const transposedLines = transpose(lines);

  // Reverse the order of lines to mirror along the x-axis
  const mirroredLines = transposedLines.reverse();

  // Join the lines back together with newline characters
  const mirroredText = mirroredLines.join('\n');

  return mirroredText;
}


function displayImages1(jsonData) {
  // Assuming just one image in the data
  const imageData = jsonData.image;
  const width = imageData.length;
  const height = imageData[0].length;
  console.log(jsonData.label);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');


  const imageDataObj = ctx.createImageData(width, height);

  let i = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = imageData[x][y];
      imageDataObj.data[i +2] = pixel[0];  // Red
      imageDataObj.data[i + 1] = pixel[1];  // Green
      imageDataObj.data[i] = pixel[2];
      imageDataObj.data[i + 3] = 255;        // Alpha (fully opaque)
      i += 4;
    }
  }


  console.log("width and height", width, " ", height);



  ctx.putImageData(imageDataObj, 0, 0);


  imageContainer.appendChild(canvas);
  let element = document.getElementById('image-container')
  element.style.rotate = "90deg";
  element.style.transform = "scaleY(-1)";


  

  // ctx.restore();


  const boxes = jsonData.box;
  const label = jsonData.label;

  let ww = boxes.length;
  for (var z = 0; z < ww; z++) {
    const [x1, y1, x2, y2] = boxes[z];
    ctx.strokeStyle = colorMap[z % 4];
    ctx.lineWidth = 3;
    console.log("before shift");
    console.log(x1, y1, x2, y2);
    ctx.strokeRect(y1, x1, (y2 - y1), (x2 - x1));
    console.log(y1, x1, y2 - y1, x2 - x1);



    const item = itemMap[(label[z])];
    console.log(x1, x2, y1, y2);
    console.log(label, item);


    if (item) {
      ctx.font = '20px Arial';
      ctx.fillStyle = colorMap[z%10];
      ctx.save();
      ctx.scale(1, -1);
      
      
      ctx.rotate(-Math.PI / 2);
  
 
      ctx.fillText(`${item}`, x1+5, y1+18);
      
     
      ctx.restore();
  } else {
      console.warn("Label not found in itemMap:", label);
  }
  


  }
  ctx.restore();
}


const imageContainer = document.getElementById('image-container');

setInterval(function () {
  const local = 'http://192.168.45.230:5000/data';
  fetch(local)
    .then(response => response.json())
    .then(jsonData => {
      imageContainer.innerHTML = ''; console.log(jsonData.boxes);
      displayImages1(jsonData)
    })
    .catch(error => console.error('Error loading JSON data:', error)
    );


}, 1000);







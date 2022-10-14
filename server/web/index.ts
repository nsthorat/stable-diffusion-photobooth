
async function main() {
  const imgElement = document.getElementById("img") as HTMLImageElement;
  const generateElement = document.getElementById("generate");
  const promptElement = document.getElementById("prompt") as HTMLTextAreaElement;

  console.log(generateElement);
  generateElement?.addEventListener("click", () => {
    const prompt = promptElement?.value;

    const context = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);

    const data = encodeURIComponent(canvas.toDataURL('image/png'));
    console.log(data);

    imgElement.src = `/generate?prompt=${prompt}&init_img=${data}`;
  });
}

// The width and height of the captured photo. We will set the
// width to the value defined here, but the height will be
// calculated based on the aspect ratio of the input stream.

const height = 128; // We will scale the photo width to this
let width = 0; // This will be computed based on the input stream

// |streaming| indicates whether or not we're currently streaming
// video from the camera. Obviously, we start at false.

let streaming = false;

// The various HTML elements we need to configure or control. These
// will be set by the startup() function.

let video = null;
let canvas = null;
let photo = null;
let startbutton = null;

function showViewLiveResultButton() {
  if (window.self !== window.top) {
    // Ensure that if our document is in a frame, we get the user
    // to first open it in its own tab or window. Otherwise, it
    // won't be able to request permission for camera access.
    document.querySelector(".contentarea").remove();
    const button = document.createElement("button");
    button.textContent = "View live result of the example code above";
    document.body.append(button);
    button.addEventListener('click', () => window.open(location.href));
    return true;
  }
  return false;
}

function startup() {
  if (showViewLiveResultButton()) { return; }
  video = document.getElementById('video');
  canvas = document.getElementById('canvas');
  photo = document.getElementById('photo');
  startbutton = document.getElementById('startbutton');

  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
      video.srcObject = stream;
      video.play();
    })
    .catch((err) => {
      console.error(`An error occurred: ${err}`);
    });

  video.addEventListener('canplay', (ev) => {
    if (!streaming) {
      //height = video.videoHeight / (video.videoWidth / width);
      width = Math.floor(video.videoWidth / (video.videoHeight / height));
      console.log(video.videoWidth);
      // Firefox currently has a bug where the height can't be read from
      // the video, so we will make assumptions if this happens.

      // if (isNaN(height)) {
      //   height = width / (4 / 3);
      // }

      video.setAttribute('width', width);
      video.setAttribute('height', height);

      canvas.setAttribute('width', width);
      canvas.setAttribute('height', height);
      streaming = true;
    }
  }, false);

}



// Set up our event listener to run the startup process
// once loading is complete.
window.addEventListener('load', startup, false);

addEventListener('DOMContentLoaded', (event) => main());

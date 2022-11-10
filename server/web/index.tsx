import * as React from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { useAppDispatch, useAppSelector } from "./hooks";
import { setMode, setPhotoModeIndex, setTimer, store } from "./store";

const TIMER_LENGTH = 1;
const NUM_PHOTOS = 3;

const WEBCAM_WIDTH = 512;
const WEBCAM_HEIGHT = 512;

const PRINTER_DPI = 300;
const PRINTER_WIDTH_INCHES = 4;
const PRINTER_HEIGHT_INCHES = 6;
const PRINTER_WIDTH_PX = 1200; // PRINTER_DPI * PRINTER_WIDTH_INCHES;
const PRINTER_HEIGHT_PX = 1800; //PRINTER_DPI * PRINTER_HEIGHT_INCHES;

export const App = React.memo(function App(): JSX.Element {
  // let body: JSX.Element;
  const dispatch = useAppDispatch();

  const webcamVideo = React.useRef<HTMLVideoElement>(null);
  const realCanvases = Array(NUM_PHOTOS)
    .fill(0)
    .map(() => React.useRef<HTMLCanvasElement>(null));
  const aiCanvases = Array(NUM_PHOTOS)
    .fill(0)
    .map(() => React.useRef<HTMLCanvasElement>(null));

  console.log(webcamVideo);
  React.useEffect(() => {
    console.log(webcamVideo.current);
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        webcamVideo.current.srcObject = stream;
        webcamVideo.current.play();
        dispatch(setMode("IDLE"));
      })
      .catch((err) => {
        console.error(`An error occurred: ${err}`);
      });
  }, []);

  const mode = useAppSelector((state) => state.app.mode);
  const timer = useAppSelector((state) => state.app.timer);
  const photoModeIndex = useAppSelector((state) => state.app.photoModeIndex);

  const start = () => {
    dispatch(setMode("PHOTO"));
    dispatch(setTimer(TIMER_LENGTH));
  };

  if (timer > 0) {
    setTimeout(() => {
      const nextTimer = timer - 1;
      if (nextTimer === 0) {
        const width = webcamVideo.current.videoWidth;
        const height = webcamVideo.current.videoHeight;
        // Capture!
        const canvasRef = realCanvases[photoModeIndex];

        console.log(canvasRef.current, photoModeIndex);
        {
          const context = canvasRef.current.getContext("2d");

          canvasRef.current.width = width;
          canvasRef.current.height = height;
          context.drawImage(webcamVideo.current, 0, 0, width, height);
        }

        // remove
        const aiCanvasRef = aiCanvases[photoModeIndex];
        {
          const context = aiCanvasRef.current.getContext("2d");
          aiCanvasRef.current.width = width;
          aiCanvasRef.current.height = height;
          context.drawImage(webcamVideo.current, 0, 0, width, height);
        }

        const nextPhotoModeIndex = photoModeIndex + 1;
        if (nextPhotoModeIndex === NUM_PHOTOS) {
          dispatch(setMode("GENERATE"));
        } else {
          dispatch(setPhotoModeIndex(photoModeIndex + 1));
          dispatch(setTimer(TIMER_LENGTH));
        }
      } else {
        dispatch(setTimer(nextTimer));
      }
    }, 1000);
  }

  let overlayContent: JSX.Element;
  if (mode === "IDLE") {
    overlayContent = (
      <button
        onClick={start}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full text-6xl"
      >
        Start!
      </button>
    );
  } else if (mode === "PHOTO") {
    overlayContent = <div className="text-white text-6xl">{timer}</div>;
  }

  let mainPanel: JSX.Element;
  if (mode === "LOADING" || mode === "IDLE" || mode === "PHOTO") {
    mainPanel = (
      <div className="w-full h-screen flex-grow video-container relative">
        <div className="video-container w-full h-screen absolute">
          <video
            width={WEBCAM_WIDTH}
            height={WEBCAM_HEIGHT}
            id="video"
            className="w-full h-screen"
            ref={webcamVideo}
          >
            Video stream not available.
          </video>
        </div>
        <div className="color-green absolute margin-auto w-full h-full grid place-items-center opacity-70 hover:opacity-100">
          {overlayContent}
        </div>
      </div>
    );
  }

  const photoRows = realCanvases.map((_, i) => {
    const realCanvas = realCanvases[i];
    const aiCanvas = aiCanvases[i];
    return (
      <>
        <div className="flex flex-row my-2">
          <div
            className="photo flex justify-center flex-0 overflow-hidden mr-1"
            key={`photo_${i}`}
          >
            <canvas
              width={WEBCAM_WIDTH}
              height={WEBCAM_HEIGHT}
              className="photo-canvas h-full"
              ref={realCanvas}
            ></canvas>
          </div>
          <div
            className="photo flex justify-center flex-0 overflow-hidden"
            key={`ai_${i}`}
          >
            <canvas
              width={WEBCAM_WIDTH}
              height={WEBCAM_HEIGHT}
              className="photo-canvas h-full"
              ref={aiCanvas}
            ></canvas>
          </div>
        </div>
        <div className="text-caption text-left pl-2 ">
          in cubism from outer space, trending on artstation, HD
        </div>
      </>
    );
  });

  const date = new Date().toLocaleString("en-us", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  return (
    <>
      <div className="flex w-full flex-row container">
        <div className="print-preview flex flex-col flex-initial">
          {/* <div className="col-preview real-preview h-full flex flex-initial flex-col flex-grow">
            {realCanvasElements}
          </div>
          <div className="col-preview ai-preview h-full flex flex-initial flex-col flex-grow">
            {aiCanvasElements}
          </div> */}
          <div className="date text-right text-xs mt-2 mr-2">{date}</div>
          {photoRows}
        </div>
        <div className="no-print main-panel relative w-full">{mainPanel}</div>
      </div>
    </>
  );
});

async function main() {
  const imgElement = document.getElementById("img") as HTMLImageElement;
  const generateElement = document.getElementById("generate");
  const promptElement = document.getElementById(
    "prompt"
  ) as HTMLTextAreaElement;

  console.log(generateElement);
  generateElement?.addEventListener("click", () => {
    const prompt = promptElement?.value;

    const context = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);

    const data = encodeURIComponent(canvas.toDataURL("image/png"));
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
    button.addEventListener("click", () => window.open(location.href));
    return true;
  }
  return false;
}

function startup() {
  if (showViewLiveResultButton()) {
    return;
  }
  video = document.getElementById("video");
  canvas = document.getElementById("canvas");
  photo = document.getElementById("photo");
  startbutton = document.getElementById("startbutton");

  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then((stream) => {
      video.srcObject = stream;
      video.play();
    })
    .catch((err) => {
      console.error(`An error occurred: ${err}`);
    });

  if (video == 1) {
    video.addEventListener(
      "canplay",
      (ev) => {
        if (!streaming) {
          //height = video.videoHeight / (video.videoWidth / width);
          width = Math.floor(video.videoWidth / (video.videoHeight / height));
          console.log(video.videoWidth);
          // Firefox currently has a bug where the height can't be read from
          // the video, so we will make assumptions if this happens.

          // if (isNaN(height)) {
          //   height = width / (4 / 3);
          // }

          video.setAttribute("width", width);
          video.setAttribute("height", height);

          canvas.setAttribute("width", width);
          canvas.setAttribute("height", height);
          streaming = true;
        }
      },
      false
    );
  }
}

// Set up our event listener to run the startup process
// once loading is complete.
//window.addEventListener("load", startup, false);

addEventListener("DOMContentLoaded", (event) => main());

window.addEventListener("DOMContentLoaded", () => {
  const root = createRoot(document.getElementById("root") as HTMLDivElement);
  root.render(
    <Provider store={store}>
      <App />
    </Provider>
  );

  //main();
});

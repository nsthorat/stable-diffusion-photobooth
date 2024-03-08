import * as React from "react";
import { useAppDispatch, useAppSelector } from "./hooks";
import PROMPTS from "./prompts.json";
import {
  setGenerateModeIndex,
  setMode,
  setPhotoModeIndex,
  setTimer,
} from "./store";

const TIMER_LENGTH = 0;
const NUM_PHOTOS = 3;
const NUM_AI_GENERATIONS = 5;
const REQUESTS_TIMEOUT = 20000; // ms

const WEBCAM_WIDTH = 512;
const WEBCAM_HEIGHT = 0;

export const Photobooth = React.memo(function App(): JSX.Element {
  const dispatch = useAppDispatch();

  const webcamVideo = React.useRef<HTMLVideoElement>(null);
  const webcamStill = React.useRef<HTMLCanvasElement>(null);
  const realCanvases = Array(NUM_PHOTOS)
    .fill(0)
    .map(() => React.useRef<HTMLCanvasElement>(null));
  const aiColumnImages = Array(NUM_PHOTOS)
    .fill(0)
    .map(() => React.useRef<HTMLImageElement>(null));
  const aiCandidateGenerations = Array(NUM_AI_GENERATIONS)
    .fill(0)
    .map(() => React.useRef<HTMLImageElement>(null));
  const aiCandidateSpinners = Array(NUM_AI_GENERATIONS)
    .fill(0)
    .map(() => React.useRef<HTMLDivElement>(null));
  const aiModeRealMainCanvas = React.useRef<HTMLCanvasElement>(null);
  const promptRef = React.useRef<HTMLTextAreaElement>(null);
  const collectImagesRef = React.useRef<HTMLInputElement>(null);

  // Local state
  const [requestsPending, setRequestsPending] = React.useState(0);
  const [showCandidateImages, setShowCandidateImages] = React.useState(false);
  const [currentPrompt, setCurrentPrompt] = React.useState("");

  const prompts = Array(NUM_PHOTOS)
    .fill(0)
    .map(() => React.useState<string>("your prompt"));

  // Setup the webcam.
  React.useEffect(() => {
    try {
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: false })
        .then((stream) => {
          webcamVideo.current.srcObject = stream;
          webcamVideo.current.play();
          webcamVideo.current.addEventListener(
            "loadeddata",
            () => {
              dispatch(setMode("IDLE"));
            },
            false
          );
        })
        .catch((err) => {
          console.error(`An error occurred: ${err}`);
        });
    } catch (e) {
      console.log(
        `Try adding ${location.host} to:`,
        "chrome://flags/#unsafely-treat-insecure-origin-as-secure"
      );
    }
  }, []);

  const mode = useAppSelector((state) => state.app.mode);
  const timer = useAppSelector((state) => state.app.timer);
  const photoModeIndex = useAppSelector((state) => state.app.photoModeIndex);
  const generateModeIndex = useAppSelector(
    (state) => state.app.generateModeIndex
  );

  const start = () => {
    dispatch(setMode("PHOTO"));
    dispatch(setTimer(TIMER_LENGTH));
  };

  if (mode === "PHOTO") {
    setTimeout(() => {
      if (timer === 0) {
        console.log(photoModeIndex);
        const nextPhotoModeIndex = photoModeIndex + 1;
        if (nextPhotoModeIndex === NUM_PHOTOS) {
          dispatch(setMode("GENERATE"));
        } else {
          dispatch(setPhotoModeIndex(photoModeIndex + 1));
          dispatch(setTimer(TIMER_LENGTH));
        }
      } else {
        dispatch(setTimer(timer - 1));
      }
    }, 1000);
    if (timer === 0) {
      const width = webcamVideo.current.videoWidth;
      const height = webcamVideo.current.videoHeight;
      // Capture!
      const canvasRef = realCanvases[photoModeIndex];
      const context = canvasRef.current.getContext("2d");
      canvasRef.current.width = width;
      canvasRef.current.height = height;
      context.drawImage(webcamVideo.current, 0, 0, width, height);

      const webcamStillContext = webcamStill.current.getContext("2d");
      webcamStill.current.style.width = webcamVideo.current.offsetWidth + "px";
      webcamStill.current.style.height =
        webcamVideo.current.offsetHeight + "px";
      webcamStill.current.width = width;
      webcamStill.current.height = height;
      webcamStillContext.drawImage(webcamVideo.current, 0, 0, width, height);
    }
  }

  let overlayContent: JSX.Element = <></>;
  if (mode === "IDLE") {
    overlayContent = (
      <button
        onClick={start}
        className="start-button text-black font-bold py-1 px-4 text-2xl h-16"
        style={{ letterSpacing: "1px" }}
      >
        START
      </button>
    );
  } else if (mode === "PHOTO") {
    overlayContent = (
      <div className="text-black timer-overlay">{timer === 0 ? "" : timer}</div>
    );
  }

  let mainPanel: JSX.Element;
  if (mode === "LOADING" || mode === "IDLE" || mode === "PHOTO") {
    mainPanel = (
      <div
        className={`w-full flex-grow video-container relative ${
          timer === 0 ? "flash" : ""
        }`}
      >
        <div className="video-container w-full absolute">
          <video
            width={WEBCAM_WIDTH}
            height={WEBCAM_HEIGHT}
            id="video"
            className={`w-full m-auto ${
              timer === 0 && mode === "PHOTO" ? "hidden" : ""
            }`}
            ref={webcamVideo}
          >
            Video stream not available.
          </video>
          <canvas
            className={`w-full m-auto video-still ${
              timer === 0 ? "" : "hidden"
            }`}
            ref={webcamStill}
          ></canvas>
        </div>
        <div className="color-green absolute margin-auto w-full h-full grid place-items-center opacity-70 hover:opacity-100">
          {overlayContent}
        </div>
      </div>
    );
  } else if (mode === "GENERATE") {
    const originalImageBase64 =
      realCanvases[generateModeIndex].current.toDataURL();

    if (aiModeRealMainCanvas.current != null) {
      aiModeRealMainCanvas.current
        .getContext("2d")
        .drawImage(realCanvases[generateModeIndex].current, 0, 0);
    }
    const generate = async (prompt: string) => {
      setShowCandidateImages(true);
      aiCandidateGenerations.forEach((aiCandidateGeneration) => {
        aiCandidateGeneration.current.src =
          "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs%3D";
      });
      console.log("generating with prompt", prompt);
      const generatePayload = {
        prompt, // currentPrompt, //promptRef.current.value,
        imageBase64: originalImageBase64,
      };

      let numRequestsOut = NUM_AI_GENERATIONS;
      setRequestsPending(NUM_AI_GENERATIONS);

      for (let i = 0; i < NUM_AI_GENERATIONS; i++) {
        aiCandidateSpinners[i].current.style.display = "block";

        fetch(`/generate_image?generate_index=${generateModeIndex}`, {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(generatePayload),
        })
          .then((response) => response.json())
          .then((content) => {
            if (content["generate_index"] != generateModeIndex) {
              return;
            }
            numRequestsOut--;
            const aiBase64 = `data:image/png;base64,${content["ai_base64"]}`;
            aiCandidateSpinners[i].current.style.display = "none";
            aiCandidateGenerations[i].current.src = aiBase64;
            setRequestsPending(numRequestsOut);
          });
      }
    };

    const aiImages = aiCandidateGenerations.map((aiCandidateGeneration, i) => {
      const selectAI = () => {
        setShowCandidateImages(false);
        setCurrentPrompt("");

        const nextGenerateModeIndex = generateModeIndex + 1;
        aiColumnImages[generateModeIndex].current.src =
          aiCandidateGeneration.current.src;
        prompts[generateModeIndex][1](promptRef.current.value);
        dispatch(setGenerateModeIndex(generateModeIndex + 1));
        aiCandidateGenerations.forEach((aiCandidateGeneration) => {
          aiCandidateGeneration.current.src =
            "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs%3D";
        });
        if (nextGenerateModeIndex === NUM_PHOTOS) {
          dispatch(setMode("PRINT"));
        }
        // reset UI.
        promptRef.current.innerText = "";
        setRequestsPending(0);
        aiCandidateSpinners.forEach((aiCandidateSpinner) => {
          aiCandidateSpinner.current.style.display = "none";
        });
      };
      const imgStyle = {
        width: realCanvases[generateModeIndex].current.offsetWidth + "px",
        height: realCanvases[generateModeIndex].current.offsetHeight + "px",
      };
      return (
        <div
          className="inline-block m-2 ai-candidate relative"
          onClick={selectAI}
        >
          <div
            className="spinner-container absolute hidden"
            ref={aiCandidateSpinners[i]}
          >
            <div className="flex justify-center items-center">
              <div
                className="
                  spinner-border
                  animate-spin
                  inline-block
                  w-8
                  h-8
                  border-4
                  rounded-full
                  text-purple-500"
                role="status"
              >
                <span className="visually-hidden"></span>
              </div>
            </div>
          </div>
          <img
            src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs%3D"
            className="ai-candidate-generation rounded"
            style={imgStyle}
            ref={aiCandidateGeneration}
          />
        </div>
      );
    });

    const existingPrompts = PROMPTS.map((prompt) => {
      const displayPrompt =
        prompt.length > 100 ? `${prompt.slice(0, 100)}...` : prompt;
      return <option value={prompt}>{displayPrompt}</option>;
    });
    const selectPrompt = (e) => {
      //promptRef.current.innerText = e.target.value;
      const prompt = e.target.value;
      setCurrentPrompt(e.target.value);
      setTimeout(() => generate(prompt));
    };
    const randomPrompt = () => {
      const prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
      console.log("random prompt is", prompt);
      //promptRef.current.innerText = prompt;
      setCurrentPrompt(prompt);
      generate(prompt);
    };

    mainPanel = (
      <div className="flex flex-col margin-auto content-center h-screen w-full overflow-hidden">
        <div className="flex flex-row justify-between prompt-top-container gap-x-8 w-full">
          <div className="">
            <img
              id="img"
              className="original-image-generation"
              src={originalImageBase64}
            />
          </div>
          <div
            className="flex flex-col gap-y-4 mx-auto"
            style={{ width: "472px" }}
          >
            <select
              onChange={selectPrompt}
              className="prompts-select w-full px-2 py-2 rounded"
              style={{ letterSpacing: "1px" }}
            >
              <option>Choose from existing prompts!</option>
              {existingPrompts}
            </select>
            <button
              onClick={randomPrompt}
              disabled={requestsPending > 0}
              id="random-prompt"
              style={{ letterSpacing: "1px" }}
              className="text-black text-left w-full generate-button font-semibold py-2 px-4 rounded shadow"
            >
              random prompt
            </button>
            <textarea
              ref={promptRef}
              id="prompt"
              className="mb-2 block p-2.5 w-full bg-white"
              style={{ letterSpacing: "1px" }}
              placeholder="Or type your prompt here..."
              rows={8}
              value={currentPrompt}
              onChange={(e) => {
                console.log("onChange");
                setCurrentPrompt(e.target.value);
              }}
            ></textarea>
            <div className="text-left">
              <button
                onClick={(e) => generate(currentPrompt)}
                disabled={requestsPending > 0}
                id="generate"
                style={{ letterSpacing: "1px" }}
                className="generate-button font-semibold py-2 px-4 text-black rounded shadow"
              >
                generate
              </button>
            </div>
          </div>
        </div>
        <div>
          <div className={`${showCandidateImages ? "" : "invisible"} pb-2`}>
            Choose your favorite image
          </div>
          <div
            className={`${
              showCandidateImages ? "" : "invisible"
            } w-128 ai-images-grid overflow-scroll -mt-2 -ml-1`}
          >
            <div className="flex flex-row items-center">
              <div>{aiImages}</div>
              <div className="">
                <button
                  className="text-6xl pl-2 text-neutral-400"
                  onClick={() => setTimeout(() => generate(currentPrompt))}
                >
                  ⟳
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  } else if (mode === "PRINT") {
    const printPage = () => {
      if (collectImagesRef.current.checked) {
        const aiImagesBase64 = aiColumnImages.map((aiColumnImage) => {
          const imgElement = aiColumnImage.current;
          const canvas = document.createElement("canvas");
          canvas.width = imgElement.naturalWidth;
          canvas.height = imgElement.naturalHeight;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(imgElement, 0, 0);
          return canvas.toDataURL();
        });
        const realImagesBase64 = realCanvases.map((realCanvas) =>
          realCanvas.current.toDataURL()
        );

        const result = [];
        for (let i = 0; i < NUM_PHOTOS; i++) {
          result.push({
            realImage: realImagesBase64[i].slice(
              "data:image/png;base64,".length
            ),
            aiImage: aiImagesBase64[i].slice("data:image/png;base64,".length),
            prompt: prompts[i],
          });
        }
        fetch(`/save_images`, {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(result),
        });
      }
      window.print();
    };
    const startAgain = () => {
      window.location.reload();
    };
    mainPanel = (
      <div className="flex flex-col items-center h-full justify-center">
        <div className="text-5xl">WOW, YOU LOOK INCREDIBLE!</div>
        <div className="text-center">
          <button
            onClick={printPage}
            id="print"
            className="bg-white hover:bg-gray-100 text-black text-3xl font-semibold py-6 px-4  rounded shadow"
          >
            Print
          </button>
          <div className="text-center text-black hidden">
            <input
              id="collect-checkbox"
              type="checkbox"
              className="mr-2 mt-4"
              ref={collectImagesRef}
              checked
            ></input>
            Allow us to collect these images for a final collage
          </div>
        </div>
        <button
          onClick={startAgain}
          id="restart"
          className="generate-button mt-12 font-semibold py-4 px-4 rounded shadow text-xl"
        >
          Restart
        </button>
      </div>
    );
  }
  const photoRows = realCanvases.map((_, i) => {
    const realCanvas = realCanvases[i];
    const aiColumnImage = aiColumnImages[i];
    return (
      <>
        <div
          className="flex flex-row justify-center gap-x-4 mx-4"
          key={`container_${i}`}
        >
          <div
            className={`${
              mode === "PHOTO" && i === photoModeIndex ? "photo-selected" : ""
            }
            photo flex justify-center flex-0 overflow-hidden mr-1`}
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
            className={`${
              mode === "GENERATE" && i === generateModeIndex
                ? "photo-selected"
                : ""
            }
             photo flex justify-center flex-0 overflow-hidden`}
            key={`ai_${i}`}
          >
            <img className="photo-canvas h-full" ref={aiColumnImage} />
          </div>
        </div>
        <div className="text-caption  px-4 py-1">{prompts[i][0]}</div>
      </>
    );
  });

  const date = new Date()
    .toLocaleString("en-us", {
      month: "numeric",
      day: "numeric",
      year: "2-digit",
    })
    .replace(/\//g, ".");

  let mainMessage = "";
  if (mode === "LOADING") {
    mainMessage = "";
  } else if (mode === "IDLE") {
    mainMessage = 'Step 1: Click "Start" and take 3 pictures';
  } else if (mode === "PHOTO") {
    const remaining = NUM_PHOTOS - photoModeIndex;
    const photosText = remaining != 1 ? "photos" : "photo";
    mainMessage = `${NUM_PHOTOS - photoModeIndex} ${photosText} remaining`;
  } else if (mode === "GENERATE") {
    mainMessage = `Step 2: Enter a prompt. Select your favorite image.`;
  } else if (mode === "PRINT") {
    mainMessage = `Step 3: Print your image! Please select the default printer.`;
  }

  return (
    <>
      <div className="flex w-screen flex-row content-container flex-wrap overflow-hidden">
        <div className="flex flex-col w-full">
          <div
            className="no-print flex flex-row flex-initial justify-between px-4 py-4"
            style={{ backgroundColor: "#262220" }}
          >
            <div
              className="print-preview-description text-4xl flex flex-row items-center gap-x-4 pl-2"
              style={{ fontFamily: "Koulen" }}
            >
              <img src="/logo.png" style={{ width: "51px" }}></img>
              AI Photobooth
            </div>
            <div className="main-panel-description text-right my-4 hidden">
              {mainMessage}
            </div>
          </div>

          <div className="main-panel-container w-full flex flex-row gap-x-8">
            <div className="no-print main-panel relative grow mt-6">
              {mainPanel}
            </div>

            <div className="flex flex-col">
              <div className="print-preview flex flex-col flex-initial mt-6 mr-12">
                <div className="photo-header text-left text-xs mx-2 flex flex-row justify-between">
                  <div className="flex flex-row items-center w-full mr-2 py-2 mx-2 mb-2">
                    <img
                      src="/logo.png"
                      style={{ height: "41px", marginLeft: "-2px" }}
                    ></img>
                    <div className="text-2xl ml-2 pt-2 grow">AI Photobooth</div>
                    <div className="justify-self-end pt-4"> {date}</div>
                  </div>
                </div>
                {photoRows}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
});

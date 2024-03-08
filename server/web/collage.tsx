import { useControls } from "leva";
import * as React from "react";
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";

export interface Visits {
  ai_image_paths: string[];
  real_image_paths: string[];
}

export const Collage = React.memo(function App(): JSX.Element {
  const [goButtonDisabled, setGoButtonDisabled] = React.useState(false);
  const [imageSourceUrl, setImageSourceUrl] = React.useState("");

  // Poll /list_visits to update the visits.
  const [visits, setVisits] = React.useState<Visits>({
    ai_image_paths: [],
    real_image_paths: [],
  });
  React.useEffect(() => {
    const interval = setInterval(() => {
      fetch("/list_visits")
        .then((response) => response.json())
        .then((data) => {
          setVisits(data);
        });
    }, 1000);
    return () => clearInterval(interval);
  }, []);
  const imagePaths = [
    ...(visits.ai_image_paths || []),
    ...(visits.real_image_paths || []),
  ];
  // console.log("visits=", imagePaths);

  const {
    targetSlices,
    sourceSlices,
    resnetLayer,
    prompt,
    upscale,
    diversity,
  } = useControls({
    targetSlices: 100,
    sourceSlices: 4,
    resnetLayer: "layer4.1.add",
    upscale: 10.0,
    diversity: 0.0,
    prompt: {
      value:
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
      rows: true,
    },
  });
  console.log(prompt);

  async function generateCollage() {
    // Make a post request to generate_collage.

    const data = {
      targetSlices,
      sourceSlices,
      resnetLayer,
      imagePaths,
      prompt,
      upscale,
      diversity,
      targetSize: [window.innerWidth, window.innerHeight],
    };
    console.log(data);

    fetch(`/generate_collage`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    }).then(async (response) => {
      const imageBytes = await response.blob();
      setImageSourceUrl(URL.createObjectURL(imageBytes));
      setGoButtonDisabled(false);
    });
    setGoButtonDisabled(true);
  }

  return (
    <>
      <div className="flex flex-col">
        <div className="flex flex-row flex-wrap w-full">
          {imageSourceUrl === "" ? (
            <div></div>
          ) : (
            <div className="m-auto">
              <TransformWrapper>
                <TransformComponent>
                  <img
                    src={imageSourceUrl}
                    style={{
                      height: window.innerHeight + "px",
                      margin: "auto",
                    }}
                  />
                </TransformComponent>
              </TransformWrapper>
            </div>
          )}
        </div>
        <div className="absolute bottom-0 right-0 w-64 mr-4 mb-4">
          <button
            onClick={generateCollage}
            disabled={goButtonDisabled}
            className="text-black w-64 bg-white hover:bg-gray-100 font-semibold py-6 px-4 border border-gray-400 rounded shadow"
          >
            {!goButtonDisabled ? "Generate!" : "Generating..."}
          </button>
        </div>
      </div>
    </>
  );
});

import * as React from "react";

export interface Visits {
  visits: string[];
}
export const Collage = React.memo(function App(): JSX.Element {
  const [visits, setVisits] = React.useState({ visits: [] });
  React.useEffect(() => {
    fetch("/list_visits")
      .then((response) => response.json())
      .then((visits: Visits) => {
        setVisits(visits);
      });
  }, []);

  const images = visits.visits.map((visit) => {
    return (
      <div>
        <div className="flex flex-row">
          <img
            className="collage-img"
            src={`/image?type=realImage&visit=${visit}`}
          ></img>
          {/* <img
            className="collage-img"
            src={`/image?type=aiImage&visit=${visit}`}
          ></img> */}
        </div>
        <div></div>
      </div>
    );
  });
  return (
    <>
      <div className="flex flex-row flex-wrap w-full">{images}</div>
    </>
  );
});

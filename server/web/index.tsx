import * as React from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Collage } from "./collage";
import { Photobooth } from "./photobooth";
import { store } from "./store";

export const App = React.memo(function App(): JSX.Element {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Photobooth />}></Route>
        <Route path="/collage" element={<Collage />}></Route>
      </Routes>
    </BrowserRouter>
  );
});

window.addEventListener("DOMContentLoaded", () => {
  const root = createRoot(document.getElementById("root") as HTMLDivElement);
  root.render(
    <Provider store={store}>
      <App />
    </Provider>
  );
});

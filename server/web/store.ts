import { configureStore, PayloadAction } from "@reduxjs/toolkit";

import { createSlice } from "@reduxjs/toolkit";
import { Mode } from "fs";

export type MODE = "LOADING" | "IDLE" | "PHOTO" | "GENERATE" | "PRINT";

export interface AppState {
  mode: Mode;
  photoModeIndex: number;
  timer: number;
  generateModeIndex: number;
}

// Define the initial state using that type
const initialState: AppState = {
  mode: "LOADING",
  photoModeIndex: 0,
  timer: 0,
  generateModeIndex: 0,
};

// NOTE: We might want to split the state into multiple slices in the future.
const appSlice = createSlice({
  name: "app",
  initialState,
  reducers: {
    setMode(state, action: PayloadAction<Mode>) {
      state.mode = action.payload;
    },
    setPhotoModeIndex(state, action: PayloadAction<number>) {
      state.photoModeIndex = action.payload;
    },
    setGenerateModeIndex(state, action: PayloadAction<number>) {
      state.generateModeIndex = action.payload;
    },
    setTimer(state, action: PayloadAction<number>) {
      state.timer = action.payload;
    },
  },
});

export const store = configureStore({
  reducer: {
    [appSlice.name]: appSlice.reducer,
  },
  devTools: process.env.NODE_ENV !== "production",
});

// Export the actions.
export const { setMode, setTimer, setPhotoModeIndex, setGenerateModeIndex } =
  appSlice.actions;

// See: https://react-redux.js.org/tutorials/typescript-quick-start
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

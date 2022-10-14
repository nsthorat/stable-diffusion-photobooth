import * as path from 'path';
import HtmlWebpackPlugin from 'html-webpack-plugin';

// We use require() here because this module has no typings.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const WriteFilePlugin = require('write-file-webpack-plugin');
import {Configuration as WebpackDevServerConfiguration} from 'webpack-dev-server';
import {Configuration as WebpackConfiguration} from 'webpack';

const WEBPACK_DEVSERVER_PORT = 9000;

const SERVER_PORT = 5000;

const NODE_MODULES = path.join(__dirname, 'node_modules');
const DIST_PATH = path.join(__dirname, 'dist');
const DIST_STATIC_PATH = path.join(DIST_PATH, 'static');

export const INDEX_HTML_OPTIONS: HtmlWebpackPlugin.Options = {
  title: 'Stable Diffusion',
  publicPath: '/',
  template: path.join(__dirname, 'server/web/index.html'),
};

export const WEBPACK_DEVSERVER_CONFIG: WebpackDevServerConfiguration = {
  port: WEBPACK_DEVSERVER_PORT,
  // Automatically refresh pages when typescript changes.
  hot: true,
  // Opens the browser window to this URL automatically when the first build completes.
  open: [`http://127.0.0.1:${SERVER_PORT}`],
  devMiddleware: {
    // Writes files to dist so we can serve them statically from express.
    writeToDisk: true,
  },
};

interface Configuration extends WebpackConfiguration {
  devServer?: WebpackDevServerConfiguration;
}

export const WEBPACK_CONFIG: Configuration = {
  mode: 'development',
  devtool: 'source-map',
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/,
        // Only compile the browser typescript code.
        include: [path.resolve('server/web')],
        use: [
          {
            loader: 'ts-loader',
          },
        ],
      },
      {
        test: /\.css$/i,
        include: [path.resolve('src/')],
        use: ['style-loader', 'css-loader', 'postcss-loader'],
      },
    ],
  },
  resolve: {
    modules: ['node_modules'],
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
  },
  entry: {
    demo: './server/web/index.ts',
  },
  output: {
    path: DIST_PATH,
    filename: 'static/bundle.js',
    hotUpdateChunkFilename: 'hot/hot-update.js',
    hotUpdateMainFilename: 'hot/hot-update.json',
  },
  plugins: [
    new HtmlWebpackPlugin(INDEX_HTML_OPTIONS),
    new WriteFilePlugin(),
  ],
  devServer: WEBPACK_DEVSERVER_CONFIG,
};

export default WEBPACK_CONFIG;

const webpack = require('webpack');
const resolve = require('path').resolve;

const config = {
 devtool: 'eval-source-map',
 entry: __dirname + '/src/index.jsx',
 output:{
      path: resolve('public'),
      filename: 'bundle.js',
      publicPath: resolve('public')
},
 resolve: {
  extensions: ['.js','.jsx','.css']
 },
 module: {
  rules: [
  {
   test: /\.jsx?/,
   loader: 'babel-loader',
   exclude: /node_modules/
  },
  {
    test: /\.css$/,
    loader: 'style-loader!css-loader?modules'
  },
  {
    test: /\.(gif|png|jpe?g|svg)$/i,
    use: [
      'file-loader',
      {
        loader: 'image-webpack-loader',
        options: {
          bypassOnDebug: true, // webpack@1.x
          disable: true, // webpack@2.x and newer
        },
      },
    ],
  },
  {
    test: /\.json$/,
    exclude: /node_modules/,
    use: [
        'file-loader?name=[name].[ext]&outputPath=portal/content/json'
    ]
  }
  ]
 }
};
module.exports = config;

const webpack = require('webpack');
const resolve = require('path').resolve;

module.exports = {
 // Crucial for development, comment out for production - using in script instead
 //  devtool: 'eval-source-map', 
 // Entry file to start bundling
 entry: __dirname + '/src/index.jsx',
 // Output for bundle
 output:{
      path: resolve('public'),
      filename: 'bundle.js',
      publicPath: resolve('public')
 },
 // Gives order of priority for extensions 
 resolve: {
  extensions: ['.js','.jsx','.css']
 },
 module: {
   // This is where you put loaders, things that let you load files in 
   // non-default ways
   // test: regex for files
   // use: here you specify the loader and its options
  rules: [
    { // Babel allows the use of React's jsx files
      test: /\.jsx?/,
      exclude: /node_modules/,
      use: {
        loader: "babel-loader",
        options: {
          presets: ['@babel/preset-env', '@babel/preset-react']
        }
      }
    },
    { // Applies css styles
      test: /\.css$/i,
      exclude: /node_modules/,
      use: ["style-loader", "css-loader"],
    },
  { // Faster loading of images
    test: /\.(gif|png|jpe?g|svg)$/i,
    exclude: /node_modules/,
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
  { // Allows the loading of static json files
    test: /\.json$/,
    exclude: /node_modules/,
    use: [
        'file-loader?name=[name].[ext]&outputPath=portal/content/json'
    ]
  }
  ]
 }
};

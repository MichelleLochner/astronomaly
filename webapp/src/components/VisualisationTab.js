import React from 'react';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import { makeStyles } from '@material-ui/core/styles';
import { blue, indigo, green } from '@material-ui/core/colors';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import {PlotImage} from './PlotImage';
import {PlotContainer} from './PlotContainer.js'
// import {
//     ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip
//   } from 'recharts';
import {XYPlot, XAxis, YAxis, MarkSeries, VerticalGridLines, HorizontalGridLines, Hint} from 'react-vis';

// Testing the scatter plot

// const myData = [{'x': '-16.820', 'y': '-43.163'}, 
// {'x': '46.367538', 'y': '-0.309925'}, 
// // {'x': '-37.459553', 'y': '-27.726826'}, 
// // {'x': '28.261280', 'y': '13.430197'}, 
// // {'x': '-16.480743', 'y': '35.427864'}, 
// // {'x': '-7.735086', 'y': '53.409462'}, 
// // {'x': '19.494431', 'y': '20.171560'}, 
// // {'x': '-23.625153', 'y': '37.620316'}, 
// // {'x': '29.343306', 'y': '-11.579516'}, 
// // {'x': '-30.035269', 'y': '-14.403445'}
// ];

// function getRandomData() {
//   return new Array(2000).fill(0).map(row => ({
//     x: Math.random() * 20,
//     y: Math.random() * 20,
//     size: Math.random() * 10,
//     color: Math.random() * 10,
//     opacity: Math.random() * 0.5 + 0.5
//   }));
// }

// const randomData = getRandomData();

/**
 * Scatter plot for visualisation
 */
class MakeScatter extends React.Component {
    constructor(props){
      super(props);
      this.state = {currentPoint:false};
    }
    render() {
        return (
            <XYPlot
              width={600}
              height={600}
              margin={{left:50, right:50, top:50, bottom:50}}
              // onMouseLeave={() => this.setState({hintValue: false})}
            >
              <VerticalGridLines />
              <HorizontalGridLines />
              <XAxis />
              <YAxis />
              <MarkSeries
                className="mark-series-example"
                sizeRange={[1, 5]}
                colorDomain = {[0, 1, 2, 3, 4, 5]}
                colorRange = {['#000004', '#2c115f', '#721f81', '#b73779', '#f1605d', '#feb078']}
                animation = {false}
                onNearestXY = {value => this.setState({currentPoint:value})}
                onValueClick={(datapoint) => this.props.dataCallback(this.state.currentPoint)}
                data={this.props.data}/>
              {/* {this.state.hintValue ? <Hint value={this.state.hintValue} /> : null} */}
            </XYPlot>

          );
    }
}

/**
 * Tab to display visualisation
 */
export class VisualisationTab extends React.Component {
    constructor(props){
      super(props);
      this.getVisualisation = this.getVisualisation.bind(this);
      this.updateDisplayData = this.updateDisplayData.bind(this);
      this.getLightCurve = this.getLightCurve.bind(this);
      this.getRawFeatures = this.getRawFeatures.bind(this);
      this.state = {data:[{x:0, y:0}], displayData:{}, light_curve_data:{}, raw_features_data:{}};
        
    }
    updateDisplayData(newData){
      if (this.props.datatype == 'image')
        this.setState({displayData:newData});
      else if (this.props.datatype == 'raw_features')
        this.setState({displayData:newData}), this.getRawFeatures(newData.id);
      else
        this.setState({displayData:newData}, this.getLightCurve(newData.id));

    }
    getVisualisation(){
        fetch("/visualisation", {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify("tsne")
        })
        // .then(res => JSON.parse(res))
        .then(res => res.json())
        .then((res) =>{
          var reformattedArray = res.map(obj =>{ 
            var rObj = {};
            const keys = Object.keys(obj)
            for (const key of keys) {
              let newval = obj[key];
              if(key !== 'id')
                newval = parseFloat(obj[key]);
              rObj[key] = newval;
            }
            return rObj;
          });
          return reformattedArray;
        })
        .then((res) => {
          this.setState({data:res})})
        .catch(console.log)
      }

    getLightCurve(original_id){
      fetch("getlightcurve", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(original_id)
      })
      .then((res) => {return res.json()})
      // .then((res)=> {console.log(res);
      //               return res})
      .then((res) => this.setState({light_curve_data:res}))
      .catch(console.log);
    }

    getRawFeatures(original_id){
      fetch("getrawfeatures", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(original_id)
      })
      .then((res) => {return res.json()})
      // .then((res)=> {console.log(res);
      //               return res})
      .then((res) => this.setState({raw_features_data:res}))
      .catch(console.log);
    }

    componentDidMount() {
      this.getVisualisation();
    }

    render() {
      // console.log('Data');
      // console.log(this.state.data);
      // console.log(myData)
        return(
            <Grid component='div' container spacing={3}>
                    <Grid item xs={12}>
                        <div></div>
                    </Grid>
                    <Grid item xs={6} container spacing={1}>
                        <MakeScatter id='scatter' data={this.state.data} dataCallback={this.updateDisplayData}/>
                    </Grid>
                    <Grid item xs={6}>
                        <h1> Object ID: {this.state.displayData.id}</h1>
                        <div>
                          <PlotContainer datatype={this.props.datatype} original_id={this.state.displayData.id} light_curve_data={this.state.light_curve_data}
                                        raw_features_data={this.state.raw_features_data}/>
                        </div>
                        <div>Anomaly Score: {this.state.displayData.color} </div>
                    </Grid>
                    <Grid item xs={12}>
                        <div></div>
                    </Grid>
            </Grid>
        )
    }
}
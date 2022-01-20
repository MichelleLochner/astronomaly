import React from 'react';
import Grid from '@material-ui/core/Grid';
import Tooltip from '@material-ui/core/Tooltip';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormControl from '@material-ui/core/FormControl';
import { MenuItem } from '@material-ui/core';
import Select from '@material-ui/core/Select';
import {PlotContainer} from './PlotContainer.js'
import Highcharts from 'highcharts'
import HighchartsMore from 'highcharts/highcharts-more';
HighchartsMore(Highcharts);
import HighchartsReact from 'highcharts-react-official'
import HighchartsColorAxis from "highcharts/modules/coloraxis"; 
HighchartsColorAxis(Highcharts);


/**
 * Scatter plot for visualisation
 */
 class MakeScatter extends React.Component {
  constructor(props){
    super(props);   
  }

  shouldComponentUpdate(newProps, newState){
    // Makes sure this component doesn't render until it has to
    if (newProps.data === this.props.data) {
      return false
    }
    else {
      return true
    }
  }

  render() {
    let callBack = this.props.callBack;
    let data_org = this.props.data;

    const options = {
      chart: {
        zoomType:'xy'
      },
      title: {
        text: ''
      },
      
      legend: {
        title: {
            text: 'Score'
        },
        layout: 'vertical',
        align: 'right',
        verticalAlign: 'middle'
    },
      xAxis: {
              title:{text:'Arbitrary units'},
              tickLength:0,
              lineColor:'transparent'
              },
      yAxis: {title:{text:'Arbitrary units'},
              gridLineWidth: 0,
              lineColor:'transparent'
              },
      colorAxis: {
              min: 0,
              max: 5,
              stops: [
                [0, '#000004'], 
                [0.2, '#2c115f'], 
                [0.4, '#721f81'], 
                [0.6, '#b73779'], 
                [0.8,'#f1605d'], 
                [1.0, '#feb078']],
              reversed: false,
              endOnTick: false,
              tickInterval: 1
      },
      tooltip: {formatter: function(){
                  return 'ID: ' + data_org[this.point.index].id + 
                          '<br>' + 'x: ' + this.point.x + 
                          '<br>' + 'y: ' + this.point.y + 
                          '<br>' + 'score: ' 
                          + data_org[this.point.index].col;
                }
              },
      credits: {enabled:false}, // Otherwise prints highcarts logo
      plotOptions: {
          series: {
              marker:{enabled: true, enabledThreshold:0, radius:5},
              
              animation: {duration:100},
              // Allows the use of objects in the data instead of only 2d arrays
              turboThreshold: 3000,
          }
      },
      series: [
        {
        name: 'transformed features',
        type:'scatter',
        data: data_org,
        colorKey: 'col',
        cursor: 'pointer',
        events: {
            click: function(event) {
              callBack(data_org[event.point.index]);
            }
          }
        }
      ]
    }
  
    return (
        <HighchartsReact
            highcharts={Highcharts}
            options={options}
            constructorType={'chart'}
            immutable={true}
          />
    )
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
      this.getAvailableColumns = this.getAvailableColumns.bind(this);
      this.handleColourBy = this.handleColourBy.bind(this);
      this.state = {
        data:[{x:0, y:0}],
        available_columns:{}, 
        colourByColumn:'score',
        displayData:{}, 
        light_curve_data:{}, 
        raw_features_data:{}};
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
          body: JSON.stringify(this.state.colourByColumn)
        })
        .then(res => res.json())
        .then((res) =>{
          var reformattedArray = res.map(obj =>{ 
            var rObj = {};
            const keys = Object.keys(obj)
            for (const key of keys) {
              let newval = obj[key];
              if(key !== 'id')
                newval = parseFloat(obj[key]);
              // Color is a special field to highcharts so we just change this key
              if(key == 'color')
                rObj['col'] = newval;
              else
              rObj[key] = newval;
            }
            return rObj;
          });
          return reformattedArray;
        })
        .then((res) => {
          this.setState({data:res})
        })
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
      .then((res) => this.setState({raw_features_data:res}))
      .catch(console.log);
    }

    getAvailableColumns(){
      fetch("/getColumns", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify("")
      })
      .then(res => res.json())
      .then((res) => {
        this.setState({available_columns:res});
      })
      .catch(console.log)
    }

    handleColourBy(e){
      const colourByColumn = e.target.value;
      this.setState({colourByColumn:colourByColumn}, this.getVisualisation);
    }

    componentDidMount() {
      this.getAvailableColumns();
      this.getVisualisation();
    }

    render() {
        // Only render this component once the data has been read in
        let scatter = <div></div>;
        let dropdown = <div></div>;
        // console.log('render')
        
        if (this.state.data.length > 1) {
          scatter =  <MakeScatter id='scatter' data={this.state.data} callBack={this.updateDisplayData}/>

          let menuItems = [];
          let column_names = this.state.available_columns;
          for (const [col, text] of Object.entries(column_names)) {
              menuItems.push(<MenuItem key={col} value={col}>{text}</MenuItem>)
          }

          let form = 
          <FormControl variant="outlined" fullWidth={true}     margin='dense'>
              <Select id="select" onChange={this.handleColourBy} value={this.state.colourByColumn}>
                {menuItems}
              </Select>
            <FormHelperText>Scoring method to colour by</FormHelperText>
          </FormControl>

          let htext = ""
          if (Object.keys(this.state.available_columns).length < 2) {
            htext = "Some options only available after running active learning" }
          
          dropdown = 
          <Tooltip title={htext} sx={{fontSize: 20}}>
              {form}
          </Tooltip>

        }
        return(
            <Grid component='div' container spacing={3} direction={'column'}>
              <Grid item lg={12} xl={12}>
                  <div></div>
              </Grid>

              <Grid item lg={12} xl={12} container direction = {'row'}>
                <Grid item lg={6} xl={4} container spacing={10} direction={'column'}>
                  <Grid item>
                      <div></div>
                  </Grid>
                  <Grid item>
                    {scatter}
                  </Grid>
                  <Grid item>
                    {dropdown}
                  </Grid>
                </Grid>

                <Grid item lg={6} xl={4} container spacing={1} direction={'column'} justifyContent={"center"}>
                  <Grid item style={{textAlign: "center"}}>
                  <h1> Object ID: {this.state.displayData.id}</h1>
                  </Grid>
                  <Grid item>
                      <PlotContainer datatype={this.props.datatype} original_id={this.state.displayData.id} light_curve_data={this.state.light_curve_data}
                                    raw_features_data={this.state.raw_features_data}/>
                  </Grid>
                  
                  <Grid item style={{textAlign: "center"}}>
                  Anomaly Score: {this.state.displayData.col}
                  </Grid>
                </Grid>
              </Grid>

              <Grid item lg={12} xl={12}>
                  <div></div>
              </Grid>
            </Grid>
        )
    }
}

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

// function getRandomData() {
//   return new Array(2000).fill(0).map(row => (
//     [Math.random() * 20, Math.random() * 20]
//   ));
// }
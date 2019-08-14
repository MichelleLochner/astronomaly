import React from 'react';
import Highcharts from 'highcharts'
import HighchartsMore from 'highcharts/highcharts-more';
HighchartsMore(Highcharts);
import HighchartsReact from 'highcharts-react-official'

export class TimeSeriesPlot extends React.PureComponent {
    constructor(props){
        super(props);
    }

    render(){
        // console.log('Called PlotLightCurve render');
      // console.log(this.props.light_curve_data);
      const options = {
        title: {
          text: ''
        },
        legend: {enabled:false},
        xAxis: {title:{text:'MJD'}},
        yAxis: {title:{text:'Magnitude'},
                reversed: true},
        credits: {enabled:false},
        plotOptions: {
            series: {
                marker:{enabled: true, enabledThreshold:0},
                animation: {duration:100}
            }
        },
        series: [
          {
          name: 'errorplot',
          type:'errorbar',
          data:this.props.light_curve_data.errors,
          whiskerLength:5,
          },
          {
          name: 'scatterplot',
          type:'scatter',
          data: this.props.light_curve_data.data,
          
        }
        ]
      }
      
      return <HighchartsReact
        highcharts={Highcharts}
        options={options}
        constructorType={'chart'}
        immutable={true}
      />
      
    }
  }
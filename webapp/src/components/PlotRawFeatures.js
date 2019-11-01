import React from 'react';
import Highcharts from 'highcharts'
import HighchartsMore from 'highcharts/highcharts-more';
HighchartsMore(Highcharts);
import HighchartsReact from 'highcharts-react-official'

export class RawFeaturesPlot extends React.PureComponent {
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
        xAxis: {
                title:{text:'Features'} ,
                categories:this.props.raw_features_data.categories
                },
        yAxis: {title:{text:''},
                reversed: false},
        credits: {enabled:false},
        plotOptions: {
            series: {
                marker:{enabled: true, enabledThreshold:0},
                animation: {duration:100}
            }
        },
        series: [
          {
          name: 'scatterplot',
          type:'scatter',
          data: this.props.raw_features_data.data,
          
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
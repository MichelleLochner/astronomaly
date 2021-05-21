import React from 'react';
import Highcharts from 'highcharts'
import HighchartsMore from 'highcharts/highcharts-more';
HighchartsMore(Highcharts);
import HighchartsReact from 'highcharts-react-official'

/**
 * Displays a time series plot
 */
export class TimeSeriesPlot extends React.PureComponent {
    constructor(props){
        super(props);
    }

    render(){
        // console.log('Called PlotLightCurve render');
      // console.log(this.props.light_curve_data);
      let data = this.props.light_curve_data.data;
      let errors = this.props.light_curve_data.errors;
      let filter_labels = this.props.light_curve_data.filter_labels;
      let filter_colors = this.props.light_curve_data.filter_colors;

      if (data == null) {
        return <div></div>
      }

      var i;
      let plot_series = [];
      for (i = 0; i < data.length; i++) {
        let error_series = {
          name: 'err_' + filter_labels[i],
          type:'errorbar',
          data:errors[i],
          whiskerLength:5,
          }

        let scatter_series = {
          name: filter_labels[i],
          type:'scatter',
          data: data[i],
        }

        if (filter_colors[i].length !== 0) {
          scatter_series['color'] = filter_colors[i]
        }

        if (errors[i].length !== 0){
          plot_series.push(error_series);}
        plot_series.push(scatter_series);
      }

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
        series: plot_series
      }
      
      return <HighchartsReact
        highcharts={Highcharts}
        options={options}
        constructorType={'chart'}
        immutable={true}
      />
      
    }
  }
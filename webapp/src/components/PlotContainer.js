import {PlotImage} from './PlotImage.js';
import {TimeSeriesPlot} from './PlotLightCurve.js';
import React from 'react';

export class PlotContainer extends React.PureComponent {

    render(){
        if (this.props.datatype == 'image')
            return <PlotImage id={this.props.original_id}/>
        else
            return <TimeSeriesPlot light_curve_data={this.props.light_curve_data}/>
       
    }
}
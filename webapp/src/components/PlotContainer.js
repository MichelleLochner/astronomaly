import {PlotImage} from './PlotImage.js';
import {TimeSeriesPlot} from './PlotLightCurve.js';
import React from 'react';
import { RawFeaturesPlot } from './PlotRawFeatures.js';

export class PlotContainer extends React.PureComponent {
    constructor(props){
        super(props);
    }

    render(){
        if (this.props.datatype == 'image')
            return <PlotImage id={this.props.original_id}/>
        else if (this.props.datatype == 'light_curve')
            return <TimeSeriesPlot light_curve_data={this.props.light_curve_data}/>
        else if (this.props.datatype == 'raw_features')
            return <RawFeaturesPlot raw_features_data={this.props.raw_features_data}/>
        else
            return <div> </div>
       
    }
}
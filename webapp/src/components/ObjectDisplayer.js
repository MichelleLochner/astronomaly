import React from 'react';
import Typography from '@material-ui/core/Typography';
import Card from '@material-ui/core/Card';
import CardHeader from '@material-ui/core/CardHeader';
import Divider from '@material-ui/core/Divider';

/**
 * Class to display arbitrary data in key pairs in a neat card
 */
export class ObjectDisplayer extends React.PureComponent{
    /**
     * You have to have a docstring here even if empty or it won't compile
     * 
     */
    constructor(props){
        super(props);
    }
    render(){
        let display_text = '';
        const keys = Object.keys(this.props.object);
        for (const key of keys){
            display_text += key +': ' + this.props.object[key]+"\n";
        }

        return(
            <Card raised={true} style={{overflow: 'auto'}}>
                <CardHeader title={this.props.title}>
                </CardHeader>     
                <Divider />     
                <Typography 
                    variant="body1" 
                    component="p" 
                    paragraph={true} 
                    overflow="visible"
                    style={{whiteSpace:"pre"}}>
                    {display_text}
                </Typography>
            </Card>
        )
    }
}
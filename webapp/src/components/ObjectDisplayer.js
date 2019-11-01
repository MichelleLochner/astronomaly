import React from 'react';
import Typography from '@material-ui/core/Typography';
import Card from '@material-ui/core/Card';

export class ObjectDisplayer extends React.PureComponent{
    render(){
        let display_text = '';
        const keys = Object.keys(this.props.object);
        for (const key of keys){
            display_text += key +': ' + this.props.object[key]+"\n";
        }

        return(
            <Card raised={true} style={{maxHeight: 200, overflow: 'auto'}}>          
                <Typography color="textSecondary" gutterBottom>
                    {this.props.title}
                </Typography>
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
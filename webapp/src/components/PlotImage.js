import React from 'react';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

/**
 * Displays the image of a particular object.
 */
export class PlotImage extends React.Component{
    constructor(props){
      super(props);
      this.getImage = this.getImage.bind(this);
      this.state = {src:""}
    }
  
    getImage() {
      fetch("getimage", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(this.props.id)
      })
      .then(res => {return res.blob()})
      .then((image) => {document.getElementById("img").src=URL.createObjectURL(image)})
      .catch(console.log);
    }
  
    render() {
      this.getImage(this.props.id);
      return (
        <div style={{width:"100%", height:"100%",display: 'flex',  justifyContent:'center', alignItems:'center'}}>
        <TransformWrapper>
          <TransformComponent>
            <img id="img" src=""/>
          </TransformComponent>
        </TransformWrapper>
        </div>
      )
    }
  }
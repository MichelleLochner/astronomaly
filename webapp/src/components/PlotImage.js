import React from 'react';
import PinchZoomPan from "react-responsive-pinch-zoom-pan";

export class PlotImage extends React.Component{
    constructor(props){
      super(props);
      this.getImage = this.getImage.bind(this);
      this.state = {src:""}
    }
  
    getImage() {
      fetch("image", {
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
        <div style={{width:"400px", height:"400px"}}>
        <PinchZoomPan position="center" maxScale={10} zoomButtons={false}>
          <img id="img" src=""/>
        </PinchZoomPan>
        </div>
      )
    }
  }
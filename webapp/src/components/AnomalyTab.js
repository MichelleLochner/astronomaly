import React from 'react';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import { makeStyles } from '@material-ui/core/styles';
import { blue, indigo, green } from '@material-ui/core/colors';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import {PlotImage} from './PlotImage.js';
import TextField from '@material-ui/core/TextField';
import {TimeSeriesPlot} from './PlotLightCurve.js';
import {ObjectDisplayer} from './ObjectDisplayer.js';
import {PlotContainer} from './PlotContainer.js'


/**
 * Tab for displaying the data, cycling through anomalous objects and adding
 * human labels.
 */
export class AnomalyTab extends React.Component {
  constructor(props){
    super(props);
    this.handleForwardBackwardClick = this.handleForwardBackwardClick.bind(this);
    this.handleForwardBackwardKey = this.handleForwardBackwardKey.bind(this);
    // this.getImage = this.getImage.bind(this);
    this.handleChangeAlgorithmClick = this.handleChangeAlgorithmClick.bind(this);
    this.changeAlgorithm = this.changeAlgorithm.bind(this);
    this.handleScoreButtonClick = this.handleScoreButtonClick.bind(this);
    this.updateOriginalID = this.updateOriginalID.bind(this);
    this.getLightCurve = this.getLightCurve.bind(this);
    this.updateObjectData = this.updateObjectData.bind(this);
    this.getMetadata = this.getMetadata.bind(this);
    this.getFeatures = this.getFeatures.bind(this);
    this.getRawFeatures = this.getRawFeatures.bind(this);
    this.handleChangeIndexChange = this.handleChangeIndexChange.bind(this);

    this.state = {id:0,
                 img_src:'',
                 original_id:'-1',
                 light_curve_data:{data:[],errors:[]},
                 raw_features_data:{data:[],categories:[]},
                 features:{},
                 metadata:{}};
    
    // this.getImage(this.state.id);
  }

  
  /**
   * Cycles on click of either the forward or back button
   * @param {event} e 
   */
  handleForwardBackwardClick(e){
    const whichButton = e.currentTarget.id;
    // console.log(whichButton);
    let newID;
    if (whichButton=="forward"){
      newID = this.state.id+1;
      /// Need some logic here checking we don't get to the end
    }
    else {
      newID = this.state.id-1;
      if (newID<0) {newID=0}
    }

    this.setState({id:newID}, this.updateOriginalID(newID));
    // this.getImage(newID);
  }

  handleForwardBackwardKey(e){
    const whichKey = e.key;

    let newID;
    if (whichKey=="ArrowRight"){
      newID = this.state.id+1;
      /// Need some logic here checking we don't get to the end
    }
    else if (whichKey=="ArrowLeft") {
      newID = this.state.id-1;
      if (newID<0) {newID=0};
    }
    else {
      newID = this.state.id;
    }

    this.setState({id:newID}, this.updateOriginalID(newID));

    e.preventDefault();
    return false;
  }

  /**
   * Called when user manually changes the index to go to a specific object.
   * @param {event} e 
   */
  handleChangeIndexChange(e){
    const newID = parseInt(e.currentTarget.value);
    this.setState({id:newID}, this.updateOriginalID(newID));
  }

  /**
   * Changes which algorithm is used to sort the objects by.
   * @param {event} e 
   */
  handleChangeAlgorithmClick(e) {
    const whichButton = e.currentTarget.id;
    // console.log(whichButton);
    this.changeAlgorithm(whichButton);
  }

  /**
   * Allows the user to label objects based on relevance
   * @param {event} e 
   */
  handleScoreButtonClick(e){
    // console.log(e.currentTarget.color);
    // e.currentTarget.color = "secondary";

    fetch("/label", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({'id':this.state.original_id, 'label':e.currentTarget.id})
    })
    .catch(console.log)

  }

  changeAlgorithm(columnName){
    fetch("/sort", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(columnName)
    })
    .then(res => res.json())
    .then((data) => {
      this.setState({id:0}, this.updateOriginalID(0));
    })
    .catch(console.log)
  }

  updateObjectData(newOriginalId){
    if (this.props.datatype=='light_curve')
      this.getLightCurve(newOriginalId);
    else if (this.props.datatype=='raw_features')
      this.getRawFeatures(newOriginalId)
    this.getFeatures(newOriginalId);
    this.getMetadata(newOriginalId);
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
    // .then((res)=> {console.log(res);
    //               return res})
    .then((res) => this.setState({light_curve_data:res}))
    .catch(console.log);
  }

  getFeatures(original_id){
    fetch("getfeatures", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(original_id)
    })
    .then((res) => {return res.json()})
    // .then((res)=> {console.log(res);
    //               return res})
    .then((res) => this.setState({features:res}))
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
    // .then((res)=> {console.log(res);
    //               return res})
    .then((res) => this.setState({raw_features_data:res}))
    .catch(console.log);
  }

  getMetadata(original_id){
    fetch("getmetadata", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(original_id)
    })
    .then((res) => {return res.json()})
    // .then((res)=> {console.log(res);
    //               return res})
    .then((res) => this.setState({metadata:res}))
    .catch(console.log);
  }

  updateOriginalID(ind){
    fetch("/getindex", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(ind)
    })
    .then(res => res.json())
    .then((res) => {
      this.setState({original_id:res}, this.updateObjectData(res));
    })
    // .then(() => console.log('getOriginalID called'))
    // .then(() => console.log(this.state.id))
    // .then(() => console.log(this.state.original_id))
    .catch(console.log)
  }

  componentDidMount(){
    if (this.state.original_id == '-1'){
      this.updateOriginalID(this.state.id);}
  }
 

  render() {
    // console.log('Anomaly')
    // console.log(this.props)
      return(
          <div>
              <Grid component='div' container spacing={3} onKeyDown={this.handleForwardBackwardKey} tabIndex="0">
                  <Grid item xs={12}>
                      <div></div>
                  </Grid>
                  <Grid item xs={8}>
                      {/* <MakePlot plot={this.props.plot}/> */}
                      <Grid container spacing={3} >
                      <Grid item xs={12} align="center">
                          <PlotContainer datatype={this.props.datatype} original_id={this.state.original_id} light_curve_data={this.state.light_curve_data}
                                        raw_features_data={this.state.raw_features_data}/>
                        </Grid>

                        <Grid item xs={12} align="center">
                          <Grid container spacing={3}>
                              <Grid item xs={4}>
                                  <Button variant="contained" align="left" id="back" onClick={this.handleForwardBackwardClick}> Back</Button>  
                              </Grid>
                              <Grid item xs={4}>
                                <TextField id="chooseNumber" label="Index" value={this.state.id} type="number" fullWidth={false} 
                                inputProps={{style:{textAlign:"center"}}} InputLabelProps={{style:{textAlign:"center"}}}
                                onChange={this.handleChangeIndexChange}/>
                              </Grid>
                              <Grid item xs={4} align="right">
                                  <Button variant="contained" align="right" id="forward" onClick={this.handleForwardBackwardClick}> Forward </Button> 
                              </Grid> 
                              

                              <Grid item xs={12} align="center">
                                  <Grid container alignItems="center">
                                      <Grid item xs={2}>
                                          <Button variant="contained" color="primary" onClick={this.handleScoreButtonClick} id="0"> 0 </Button> 
                                      </Grid> 
                                      <Grid item xs={2}>
                                          <Button variant="contained" color="primary"onClick={this.handleScoreButtonClick} id="1"> 1 </Button> 
                                      </Grid> 
                                      <Grid item xs={2}>
                                          <Button variant="contained" color="primary"onClick={this.handleScoreButtonClick} id="2"> 2 </Button> 
                                      </Grid> 
                                      <Grid item xs={2}>
                                          <Button variant="contained" color="primary"onClick={this.handleScoreButtonClick} id="3"> 3 </Button> 
                                      </Grid> 
                                      <Grid item xs={2}>
                                          <Button variant="contained" color="primary"onClick={this.handleScoreButtonClick} id="4"> 4 </Button> 
                                      </Grid> 
                                      <Grid item xs={2}>
                                          <Button variant="contained" color="primary"onClick={this.handleScoreButtonClick} id="5"> 5 </Button> 
                                      </Grid> 
                                  </Grid>
                              </Grid>

                              <Grid item xs={12} align="center">
                                <Grid container alignItems="center">
                                      <Grid item xs={4}>
                                          <Button variant="contained" id="random" onClick={this.handleChangeAlgorithmClick}> Random </Button> 
                                      </Grid> 
                                      <Grid item xs={4}>
                                          <Button variant="contained" id="score" onClick={this.handleChangeAlgorithmClick}> ML Algorithm </Button> 
                                      </Grid> 
                                      <Grid item xs={4}>
                                          <Button variant="contained" id="final_score" onClick={this.handleChangeAlgorithmClick}> Trained </Button> 
                                      </Grid> 
                                  </Grid>

                              </Grid>
                            </Grid>  
                        </Grid>
                      </Grid>
                  </Grid>

                  <Grid item xs={2}>
                    <Grid container alignItems="center" spacing={5}>
                      <Grid item xs={12}>
                          <ObjectDisplayer title='Metadata' object={this.state.metadata} />
                      </Grid>

                      <Grid item xs={12}>
                        <ObjectDisplayer title='Features' object={this.state.features} />
                      </Grid>
                    </Grid>
                  </Grid>
                  <Grid item xs={12}>
                      <div></div>
                  </Grid>




              </Grid>

          </div>
      )
    }

}
import React from 'react';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
// import {makeStyles } from '@material-ui/core/styles';
import {createMuiTheme, MuiThemeProvider } from '@material-ui/core/styles'
import { blue, indigo, green, grey } from '@material-ui/core/colors';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import {PlotImage} from './PlotImage.js';
import TextField from '@material-ui/core/TextField';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormControl from '@material-ui/core/FormControl';
import {TimeSeriesPlot} from './PlotLightCurve.js';
import {ObjectDisplayer} from './ObjectDisplayer.js';
import {PlotContainer} from './PlotContainer.js'
import { MenuItem, Icon } from '@material-ui/core';
import Divider from '@material-ui/core/Divider';
import CircularProgress from '@material-ui/core/CircularProgress';
import IconButton from '@material-ui/core/IconButton';
import SkipNext from '@material-ui/icons/SkipNext';
import SkipPrevious from '@material-ui/icons/SkipPrevious';

const muiTheme = createMuiTheme({ palette: {primary: {main:grey[300]},
                                            secondary:{main:indigo[500]} }})
                                            

/**
 * Tab for displaying the data, cycling through anomalous objects and adding
 * human labels.
 */
export class AnomalyTab extends React.Component {
  constructor(props){
    super(props);
    this.handleForwardBackwardClick = this.handleForwardBackwardClick.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleSortBy = this.handleSortBy.bind(this);
    this.changeSortBy = this.changeSortBy.bind(this);
    this.handleRetrainButton = this.handleRetrainButton.bind(this);
    this.handleScoreButtonClick = this.handleScoreButtonClick.bind(this);
    this.updateOriginalID = this.updateOriginalID.bind(this);
    this.getLightCurve = this.getLightCurve.bind(this);
    this.updateObjectData = this.updateObjectData.bind(this);
    this.getMetadata = this.getMetadata.bind(this);
    this.getFeatures = this.getFeatures.bind(this);
    this.getRawFeatures = this.getRawFeatures.bind(this);
    this.handleChangeIndexChange = this.handleChangeIndexChange.bind(this);
    this.doNothing = this.doNothing.bind(this);
    this.changeButtonColor = this.changeButtonColor.bind(this);
    this.getMaxID = this.getMaxID.bind(this);

    this.state = {id:0,
                 max_id:0,
                 img_src:'',
                 original_id:'-1',
                 light_curve_data:{data:[],errors:[]},
                 raw_features_data:{data:[],categories:[]},
                 features:{},
                 metadata:{},
                 button_colors:{"0": "primary",
                                "1": "primary",
                                "2": "primary",
                                "3": "primary",
                                "4": "primary",
                                "5": "primary"},
                 sortby:"score",
                 training: false};
    
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
      if (newID >= this.state.max_id) {newID = this.state.max_id - 1}
    }
    else {
      newID = this.state.id-1;
      if (newID<0) {newID=0}
    }

    this.setState({id:newID}, this.updateOriginalID(newID));
    // this.getImage(newID);
  }

  handleKeyDown(e){
    const whichKey = e.key;
    const allowed_keys = ["0", "1", "2", "3", "4", "5"];

    let newID;
    if (whichKey=="ArrowRight"){
      newID = this.state.id+1;
      if (newID >= this.state.max_id) {newID = this.state.max_id - 1}
      e.preventDefault();
      
    }
    else if (whichKey=="ArrowLeft") {
      newID = this.state.id-1;
      if (newID<0) {newID=0};
      e.preventDefault();
    }
    else {
      newID = this.state.id;
      
      if ((isNaN(whichKey) == false) && (allowed_keys.includes(whichKey))){
        this.changeButtonColor(whichKey);
        fetch("/label", {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({'id':this.state.original_id, 'label':whichKey})
        })
        .catch(console.log)

        e.preventDefault();
      }
    }

    this.setState({id:newID}, this.updateOriginalID(newID));

    return false;
  }

  /**
   * Called when user manually changes the index to go to a specific object.
   * @param {event} e 
   */
  handleChangeIndexChange(e){
    const value = e.currentTarget.value;
    if (isNaN(value) === false) {
      let newID = parseInt(value);
      if (newID >= this.state.max_id) {newID = this.state.max_id - 1}
      this.setState({id:newID}, this.updateOriginalID(newID));
      e.stopPropagation();
    }
  }

  doNothing(e) {
    e.stopPropagation();
  }

  /**
   * Changes which scores are used to sort the objects by.
   * @param {event} e 
   */
  handleSortBy(e){
    const sortByColumn = e.target.value;
    this.setState({sortby:sortByColumn});
    this.changeSortBy(sortByColumn);
  }

  handleRetrainButton(e) {
    this.setState({training:true})
    fetch("/retrain", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify('retrain')
    })
    .then((res) => {this.setState({sortby:"trained_score", training:false});
                    this.changeSortBy("trained_score")})
    .catch(console.log)
  }

  /**
   * Allows the user to label objects based on relevance
   * @param {event} e 
   */
  changeButtonColor(button_id) {
    let new_colors = this.state.button_colors;
    for (const key in new_colors) {
      new_colors[key] = "primary";
    }
    if (button_id !== "-1") {
      new_colors[button_id] = "secondary";
    }
    this.setState({button_colors: new_colors});
  }

  resetButtonColors() {
    let new_colors = this.state.button_colors;
    for (const key in new_colors) {
      new_colors[key] = "primary";
    }
    this.setState({button_colors: new_colors});
  }
   handleScoreButtonClick(e){
    // console.log(e.currentTarget.color);
    // e.currentTarget.color = "secondary";
    
    this.changeButtonColor(e.currentTarget.id);
    fetch("/label", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({'id':this.state.original_id, 'label':e.currentTarget.id})
    })
    .then((res) => {this.getMetadata(this.state.original_id)})
    .catch(console.log)

  }

  /**
   * Tells the backend to reorder the data according to a different scoring 
   * method
   * @param {string} columnName
   */
  changeSortBy(columnName){
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
    .then((res) => {this.changeButtonColor(parseInt(this.state.metadata.human_label).toString())})
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

  getMaxID() {
    fetch("/getmaxid", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify("")
    })
    .then(res => res.json())
    .then((res) => {
      this.setState({max_id:parseInt(res)});
    })
    .catch(console.log)
  }

  componentDidMount(){
    if (this.state.original_id == '-1'){
      this.updateOriginalID(this.state.id);}
    if (this.state.max_id == 0) {
      this.getMaxID();
    }
  }
 

  render() {
    // console.log('Anomaly')
    // console.log(this.props)
      return(
          <div>
              <Grid component='div' container spacing={3} onKeyDown={this.handleKeyDown} tabIndex="0">
                  <Grid item xs={12}></Grid>
                  <Grid item container xs={8} justify="center">
                      {/* <MakePlot plot={this.props.plot}/> */}
                      <Grid container spacing={3} alignItems="center">
                        <Grid item xs={12} align="center">
                          <PlotContainer datatype={this.props.datatype} original_id={this.state.original_id} light_curve_data={this.state.light_curve_data}
                                        raw_features_data={this.state.raw_features_data}/>
                        </Grid>

                        <Grid container item xs={12} align="center">
                          <Grid item xs={2}>
                            {/* <Button variant="contained" id="back" startIcon={<SkipPrevious />} onClick={this.handleForwardBackwardClick}>Back</Button>   */}
                            <IconButton id="back" size="medium" onClick={this.handleForwardBackwardClick}>
                              <SkipPrevious />
                            </IconButton>
                          </Grid>

                          <Grid item xs={3}>
                          </Grid>
                          <Grid item xs={2}>
                            <TextField id="chooseNumber" value={this.state.id} type="number" fullWidth={false} 
                            inputProps={{style:{textAlign:"center"}}} FormHelperTextProps={{style:{textAlign:"center"}}}
                            onChange={this.handleChangeIndexChange} onKeyDown={this.doNothing} helperText="Index" />
                          </Grid>
                          <Grid item xs={3}>
                          </Grid>
                          <Grid item xs={2} align="center">
                            {/* <Button variant="contained" id="forward" onClick={this.handleForwardBackwardClick}> F </Button>  */}
                            <IconButton id="forward" size="medium" onClick={this.handleForwardBackwardClick}>
                              <SkipNext />
                            </IconButton>
                        </Grid>
                      </Grid>
                              
                        <Grid item xs={12} container justify="center">

                          <Grid item xs={4}>
                            <Typography variant="overline" display="block">
                              How interesting is this object?
                            </Typography>
                          </Grid>

                        </Grid>

                        <Grid item xs={12} align="center">
                            <Grid container alignItems="center">
                            <MuiThemeProvider theme={muiTheme}>
                                <Grid item xs={2}>
                                    <Button variant="contained" color={this.state.button_colors["0"]} onClick={this.handleScoreButtonClick} id="0"> 0 </Button>  
                                </Grid> 
                                <Grid item xs={2}>
                                    <Button variant="contained" color={this.state.button_colors["1"]} onClick={this.handleScoreButtonClick} id="1"> 1 </Button> 
                                </Grid> 
                                <Grid item xs={2}>
                                    <Button variant="contained" color={this.state.button_colors["2"]} onClick={this.handleScoreButtonClick} id="2"> 2 </Button> 
                                </Grid> 
                                <Grid item xs={2}>
                                    <Button variant="contained" color={this.state.button_colors["3"]} onClick={this.handleScoreButtonClick} id="3"> 3 </Button> 
                                </Grid> 
                                <Grid item xs={2}>
                                    <Button variant="contained" color={this.state.button_colors["4"]} onClick={this.handleScoreButtonClick} id="4"> 4 </Button> 
                                </Grid> 
                                <Grid item xs={2}>
                                    <Button variant="contained" color={this.state.button_colors["5"]} onClick={this.handleScoreButtonClick} id="5"> 5 </Button> 
                                </Grid> 
                                </MuiThemeProvider>
                            </Grid>
                            
                        </Grid>

                        <Grid item xs={12} container justify="center" alignItems="center">
                          {/* <Grid container alignItems="center"> */}
                            {/* <Grid item xs={1}>
                            </Grid> */}
                            <Grid item xs={6}>
                              <Grid container item xs={12} justify="center">
                                <Grid item xs={8}>
                                <FormControl variant="outlined" fullWidth={true} margin='dense'>
                                  {/* <InputLabel id="select-label" margin="dense">Sort By</InputLabel> */}
                                  <Select id="select" onChange={this.handleSortBy} value={this.state.sortby}>
                                    <MenuItem value="score">Raw anomaly score </MenuItem>
                                    <MenuItem value="trained_score">Human retrained score</MenuItem>
                                    <MenuItem value="random">Random</MenuItem>
                                  </Select>
                                  <FormHelperText>Scoring method to sort by</FormHelperText>
                                </FormControl>
                                </Grid>
                                <Grid item xs={2}></Grid>
                              </Grid>
                            </Grid>
                            <Grid item xs={2}></Grid>
                            <Grid item xs={2}>
                            {this.state.training && <CircularProgress/>}
                            </Grid>
                            
                            <Grid item xs={2}>
                              <Button variant="contained" color="primary" id="retrain" onClick={this.handleRetrainButton} disabled={this.state.training}> 
                                Retrain
                              </Button> 
                            </Grid>
                          </Grid>
                        {/* </Grid> */}
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
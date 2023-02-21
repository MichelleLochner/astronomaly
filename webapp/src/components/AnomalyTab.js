import React from 'react';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import {createTheme, MuiThemeProvider } from '@material-ui/core/styles'
import { blue, indigo, green, grey } from '@material-ui/core/colors';
import Typography from '@material-ui/core/Typography';
import TextField from '@material-ui/core/TextField';
import Select from '@material-ui/core/Select';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormControl from '@material-ui/core/FormControl';
import Switch from '@material-ui/core/Switch';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import { MenuItem } from '@material-ui/core';
import {ObjectDisplayer} from './ObjectDisplayer.js';
import {PlotContainer} from './PlotContainer.js'
import CircularProgress from '@material-ui/core/CircularProgress';
import IconButton from '@material-ui/core/IconButton';
import SkipNext from '@material-ui/icons/SkipNext';
import SkipPrevious from '@material-ui/icons/SkipPrevious';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import Tooltip from '@material-ui/core/Tooltip';

const muiTheme = createTheme({ palette: {primary: {main:grey[300]},
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
    this.handleUnlabelledSwitch = this.handleUnlabelledSwitch.bind(this);
    this.getAvailableColumns = this.getAvailableColumns.bind(this);
    this.handleRetrainButton = this.handleRetrainButton.bind(this);
    this.handleViewerButton = this.handleViewerButton.bind(this);
    this.handleDeleteLabelsButton = this.handleDeleteLabelsButton.bind(this);
    this.handleDialogClose = this.handleDialogClose.bind(this);
    this.handleScoreButtonClick = this.handleScoreButtonClick.bind(this);
    this.updateOriginalID = this.updateOriginalID.bind(this);
    this.getLightCurve = this.getLightCurve.bind(this);
    this.updateObjectData = this.updateObjectData.bind(this);
    this.getMetadata = this.getMetadata.bind(this);
    this.getCoordinates = this.getCoordinates.bind(this);
    this.getFeatures = this.getFeatures.bind(this);
    this.getRawFeatures = this.getRawFeatures.bind(this);
    this.handleChangeIndexChange = this.handleChangeIndexChange.bind(this);
    this.doNothing = this.doNothing.bind(this);
    this.changeButtonColor = this.changeButtonColor.bind(this);
    this.getMaxID = this.getMaxID.bind(this);
    this.getCurrentListIndex = this.getCurrentListIndex.bind(this);
    this.setCurrentListIndex = this.setCurrentListIndex.bind(this);

    this.state = {id:-1,
                 max_id:0,
                 img_src:'',
                 original_id:'-1',
                 light_curve_data:{data:[],errors:[]},
                 raw_features_data:{data:[],categories:[]},
                 features:{},
                 metadata:{},
                 search_cds:'',
                 search_das:'',
                 using_fits:false,
                 button_colors:{"0": "primary",
                                "1": "primary",
                                "2": "primary",
                                "3": "primary",
                                "4": "primary",
                                "5": "primary"},
                 available_columns:{},
                 sortby:"score",
                 unlabelled_first: false,
                 training: false,
                 dialog_open: false};
    
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
    this.changeSortBy(sortByColumn, this.state.unlabelled_first);
  }

  handleUnlabelledSwitch(e){
    const unlabelled_first = e.target.checked;
    this.setState({unlabelled_first:unlabelled_first});
    this.changeSortBy(this.state.sortby, unlabelled_first);
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
    .then((res) => {return res.json()})
    .then((res) => {
      if (res == "success") {
        this.setState({sortby:"trained_score", training:false});
        this.getAvailableColumns();
        this.changeSortBy("trained_score")
      }
      else {
        this.setState({training:false})
      }
    })
    .catch(console.log)
  }

  checkFitsFile(e) {
    fetch("/checkFits", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify("")
    })
    .then(res => res.json())
    .then((res) => {
      this.setState({using_fits:res});
    })
    .catch(console.log)
  }

  handleViewerButton(e) {
    fetch("/openViewer", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(this.state.original_id)
    })
    .then((res) => {return res.json()})
    .catch(console.log)
  }

  handleDeleteLabelsButton(e) {
    this.setState({dialog_open:true});
  }

  handleDialogClose(e) {
    if (e.currentTarget.id == "dialog_yes") {
      fetch("/deletelabels", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify('deletelabels')
      })
      .then((res) => {this.setState({sortby:"score", dialog_open:false})})
      .then((res) => this.changeSortBy("score"))
      .then((res) => this.resetButtonColors())
      .catch(console.log)}
    else {
      this.setState({dialog_open:false})}
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
   * Gets available columns to sort the data by
   */
  getAvailableColumns(){
    fetch("/getColumns", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify("")
    })
    .then(res => res.json())
    .then((res) => {
      this.setState({available_columns:res});
    })
    .catch(console.log)
  }
  /**
   * Tells the backend to reorder the data according to a different scoring 
   * method
   * @param {string} columnName
   */
  changeSortBy(columnName, unlabelled_first){
    let args = [columnName, unlabelled_first];
    fetch("/sort", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(args)
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
    this.getCoordinates(newOriginalId);
    this.setCurrentListIndex();
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

  getCoordinates(original_id){
    fetch("getcoordinates", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(original_id)
    })
    .then((res) => {return res.json()})
    .then((res) => {
      if(res.ra!=undefined){
        let search_cds = "http://cdsportal.u-strasbg.fr/?target=" + 
                  res.ra + '%2C' + res.dec
        let search_das = "https://das.datacentral.org.au/das?RA=" + 
                  res.ra + '&DEC=' + res.dec +"&FOV=2.0&ERR=10.0&CAT=0"
        this.setState({search_cds:search_cds, search_das:search_das})
      }
    })
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

  getCurrentListIndex(){
    fetch("/getlistindex", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify("")
    })
    .then(res => res.json())
    .then((res) => {
      this.setState({id:parseInt(res)}, this.updateOriginalID(res));
    })
    .catch(console.log)
  }

  setCurrentListIndex(){
    fetch("/setlistindex", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(this.state.id)
    })
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
    if (this.state.id == -1){
      this.getCurrentListIndex();
    }
    else if (this.state.original_id == '-1'){
      this.updateOriginalID(this.state.id);}
    if (this.state.max_id == 0) {
      this.getMaxID();
    }
    this.getAvailableColumns();
    this.checkFitsFile();
    this.changeSortBy(this.state.sortby, this.state.unlabelled_first);
  }
 

  render() {
      // Sort by menu
      let menuItems = [];
      let column_names = this.state.available_columns;
      for (const [col, text] of Object.entries(column_names)) {
          menuItems.push(<MenuItem key={col} value={col}>{text}</MenuItem>)
      }
      // Manually add the Random option
      menuItems.push(<MenuItem key={'random'} value={'random'}>{"Random"}</MenuItem>)

      let sort_by_form = 
      <FormControl variant="outlined" fullWidth={true}     margin='dense'>
          <Select id="select" onChange={this.handleSortBy} value={this.state.sortby}>
            {menuItems}
          </Select>
        <FormHelperText>Scoring method to sort by</FormHelperText>
      </FormControl>

      return(
              <Grid component='div' container spacing={3} onKeyDown={this.handleKeyDown} tabIndex="0">
                  <Grid item xs={12}></Grid>
                  <Grid item container xs={6} justifyContent="center" >
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
                              
                        <Grid item xs={12} container justifyContent="center">

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

                        <Grid item xs={12} container justifyContent="center" alignItems="center">
                          {/* <Grid container alignItems="center"> */}
                            {/* <Grid item xs={1}>
                            </Grid> */}
                            <Grid item xs={4}>
                              <Grid container item xs={12} justifyContent="center">
                                <Grid item xs={8}>
                                  {sort_by_form}
                                </Grid>
                                <Grid item xs={2}></Grid>
                              </Grid>
                            </Grid>

                            <Grid item xs={4}>
                            <Tooltip 
                              title={<Typography>Pushes all labelled data to the end of the list, prioritising the objects you have not yet seen</Typography>} placement="top">
                                <FormGroup>
                                  <FormControlLabel control={
                                    <Switch color="primary"
                                    checked={this.state.unlabelled_first}
                                    onChange={this.handleUnlabelledSwitch}/>} 
                                  label="Show unlabelled objects first" />
                                </FormGroup>
                              </Tooltip>
                            </Grid>

                            <Grid item xs={2}>
                            {this.state.training && <CircularProgress/>}
                            {!this.state.training && 
                              <Button variant="contained" color="primary" id="delete_labels" onClick={this.handleDeleteLabelsButton}> 
                                Delete Labels
                              </Button> }
                              <Dialog
                                open={this.state.dialog_open}
                                onClose={this.handleDialogClose}
                                aria-labelledby="alert-dialog-title"
                                aria-describedby="alert-dialog-description"
                              >
                                <DialogTitle id="alert-dialog-title">{"Delete all user labels?"}</DialogTitle>
                                <DialogContent>
                                  <DialogContentText id="alert-dialog-description">
                                    This will permanently delete all the labels that you may have painstakingly applied to the data.
                                    Are you sure you want to do this?
                                  </DialogContentText>
                                </DialogContent>
                                <DialogActions>
                                  <Button onClick={this.handleDialogClose} id="dialog_no" color="primary" autoFocus>
                                    No
                                  </Button>
                                  <Button onClick={this.handleDialogClose} id="dialog_yes" color="primary">
                                    Yes
                                  </Button>
                                </DialogActions>
                              </Dialog>


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
                  

                  <Grid item xs={4}>
                    <Grid container alignItems="flex-start" spacing={5}>
                      <Grid item xs={8}>
                          <ObjectDisplayer title='Metadata' object={this.state.metadata} />
                      </Grid>

                      <Grid item xs={8}>
                        <ObjectDisplayer title='Features' object={this.state.features} />
                      </Grid>
                      <Grid item xs={8} >
                        {(this.state.search_cds.length >0) &&
                        // Only display if there are coordinates to search by
                          <Tooltip title={<Typography>Opens the CDS portal to search for this object in other datasets</Typography>}>
                            <Button variant="contained" color="primary" id="search1" href={this.state.search_cds} target="_blank">
                              Search by Coordinates (CDS)
                            </Button> 
                          </Tooltip>
                        }
                      </Grid>
                        
                      <Grid item xs={8} >
                        {(this.state.search_das.length >0) &&
                          // Only display if there are coordinates to search by
                          <Tooltip title={<Typography>Opens DAS which contains extra datasets but requires login</Typography>}>
                            <Button variant="contained" color="primary" id="search2" href={this.state.search_das} target="_blank">
                              Search by Coordinates (DAS)
                            </Button> 
                          </Tooltip>
                        }
                      </Grid>

                      <Grid item xs={8} >
                        {(this.state.using_fits) &&
                          // Only display if these are fits files
                         <Tooltip title={<Typography>Opens the image in your system's local fits viewer (you can set the command in your script)</Typography>}>
                            <Button variant="contained" color="primary" id="fits_viewer_button" onClick={this.handleViewerButton}>
                              Open with Local Viewer
                            </Button> 
                          </Tooltip>
                        }
                      </Grid>

                    </Grid>
                  </Grid>
                  <Grid item xs={12}>
                  </Grid>




              </Grid>
      )
    }

}
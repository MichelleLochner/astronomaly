import React from 'react';
import IconButton from '@material-ui/core/IconButton';
import Cancel from '@material-ui/icons/Cancel';
import Tooltip from '@material-ui/core/Tooltip';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography'
import { ThemeProvider } from '@material-ui/styles';
import { withStyles } from '@material-ui/styles';
import { createTheme } from '@material-ui/core/styles'
import { blue, indigo} from '@material-ui/core/colors'
import AppBar from '@material-ui/core/AppBar';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import './App.css';
import {AnomalyTab} from './components/AnomalyTab'
import {VisualisationTab} from './components/VisualisationTab'

// const DATA_TYPE = 'image';
// const DATA_TYPE = 'light_curve';

const styles = theme =>({
  root: {
    // backgroundColor: 'white',
    // width: 500,
    // flexGrow: 1,
  },
  card: {
    minWidth: 10,
    maxWidth: 400,
    backgroundColor: blue[100]
    
  },
  bullet: {
    display: 'inline-block',
    margin: '0 2px',
    transform: 'scale(0.8)',
  },
  title: {
    fontSize: 14,
  },
  pos: {
    marginBottom: 12,
  },
  palette: {
    secondary: {
      main: blue[900]
    },
    primary: {
      main: indigo[700]
    }
  },
  typography: {
    // Use the system font instead of the default Roboto font.
    fontFamily: [
      '"Lato"',
      'sans-serif'
    ].join(',')
  }
});

const theme = createTheme({
  palette: {
    secondary: {
      main: blue[400]
    },
    primary: {
      main: indigo[700]
    },
    background: {
      main: blue[100]}
  },
});

/**
 * Contains the tabs
 * 
 * @param {Params} data - blah 
 */
function TabContainer({ children, dir }) {
  return (
    <Typography component="div" dir={dir} style={{ padding: 8 * 3 }}>
      {children}
    </Typography>
  );
}

/**
 * The App class
 */
class App extends React.Component {
  constructor(props){
    super(props);
    this.handleChange = this.handleChange.bind(this);
    this.getDataType = this.getDataType.bind(this);
    this.closeWindow = this.closeWindow.bind(this);
    this.state = {tabNumber: 0,
                  dataType: '',
                  closing: false};
  }



  handleChange(event, newValue) {
    this.setState({tabNumber: newValue});
  }

  getDataType(){
    fetch("getdatatype", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify("datatype")
    })
    .then((res) => {return res.json()})
    // .then((res)=> {console.log(res);
    //               return res})
    .then((res) => this.setState({dataType:res}))
    .catch(console.log);
  }

  closeWindow(){
    this.setState({closing:true})
    fetch("close", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify("")
    })
    .catch(console.log)   
  }

  componentDidMount(){
    this.getDataType();
  }

  render (){
    // console.log('main app rendering')
    // console.log(this.props);
    if (this.state.closing) {
      return (
          <Grid container spacing={3}>
            <Grid item xs={12}> </Grid>
            <Grid item xs={12} align="center">
              <Typography variant="h5">
                Astronomaly has been shut down. You may now close this window.
              </Typography>
            </Grid>
          </Grid>
      )
    }
    
    const { classes } = this.props;
    return (
      <div className={classes.root}>
        <ThemeProvider theme={theme}>
          <AppBar position="static" color="default">
            <Grid container>
              <Grid item xs={8}>
                <Tabs 
                  value={this.state.tabNumber}
                  onChange={this.handleChange}
                  indicatorColor="primary"
                  textColor="primary"
                  centered
                >
                  {/* <Tab label="Algorithm" /> */}
                  <Tab label="Anomaly Scoring"/>
                  <Tab label="Visualisation"/>
                </Tabs>
              </Grid>
              <Grid item xs={2}>
                {/* <Button>Close</Button> */}
                <Tooltip title="Close Astronomaly">
                  <IconButton id="close" size="medium" onClick={this.closeWindow}>
                    <Cancel />
                  </IconButton>
                </Tooltip>
              </Grid>
            </Grid>
          </AppBar>

          {/* {this.state.tabNumber === 0 && <AlgorithmTab />} */}
          {this.state.tabNumber === 0 && <AnomalyTab datatype={this.state.dataType} />}
          {this.state.tabNumber === 1 && <VisualisationTab datatype={this.state.dataType}/>}
          {/* {this.state.tabNumber === 1 && <AnomalyTab datatype={this.state.dataType} />} */}
        </ThemeProvider>
      </div>
    );
  }
}

/**
 * Main App class
 */
export default withStyles(styles)(App);



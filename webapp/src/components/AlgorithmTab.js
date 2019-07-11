import React from 'react';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import { makeStyles } from '@material-ui/core/styles';
import { blue, indigo, green } from '@material-ui/core/colors'
import Typography from '@material-ui/core/Typography'
import Paper from '@material-ui/core/Paper'


export class AlgorithmTab extends React.Component {
    

    render(){
        return(
            <div>

                <Grid component='span' container spacing={3} align='center' justify='center'>
                <Grid item xs={12}>
                        <div></div>
                    </Grid>
                        <Grid item xs={3}>
                            <Button variant="contained" align='center' >Select Python Script</Button>
                        </Grid>
                        <Grid item xs={3}>
                            <Button variant="contained" align='center'>Reload script</Button>
                        </Grid>
                        <Grid item xs={12}>
                        <Typography variant="h6" component="h2" align={'center'}>
                            Using Python script blah.py
                        </Typography>
                        </Grid>


                    <Grid item xs={12}>
                        <div></div>
                    </Grid>
                    <Grid item xs={3}>
                        <Button variant="contained">Select data</Button>
                    </Grid>
                    <Grid item xs={12}>
                    <Typography variant="h6" component="h2" align={'center'}>
                        Using data from source:
                    </Typography>
                    </Grid>
                    <Grid item xs={12}>
                        <div></div>
                    </Grid>
                    <Grid item xs={3}>
                        <Button variant="contained">Run script</Button>
                    </Grid>
                    <Grid item xs={12}>
                    <Typography variant="body1" component="span" align={'center'}>
                        Console output
                    </Typography>
                    </Grid>

                </Grid>

            </div>
        )
    }

}
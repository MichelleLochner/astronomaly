import React from 'react';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import { makeStyles } from '@material-ui/core/styles';
import { blue, indigo, green } from '@material-ui/core/colors';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import {
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip
  } from 'recharts';

const data = [
    { x: 100, y: 200, z: 200 },
    { x: 120, y: 100, z: 260 },
    { x: 170, y: 300, z: 400 },
    { x: 140, y: 250, z: 280 },
    { x: 150, y: 400, z: 500 },
    { x: 110, y: 280, z: 200 },
  ];

class MakeScatter extends React.Component {
    render() {
        return(
            <ScatterChart
            width={700}
            height={400}
            margin={{
              top: 20, right: 20, bottom: 20, left: 20,
            }}
          >
            <CartesianGrid />
            <XAxis type="number" dataKey="x" name="stature" unit="cm" />
            <YAxis type="number" dataKey="y" name="weight" unit="kg" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="A school" data={data} fill="#8884d8" />
          </ScatterChart>
        )
    }
}

export class ClusteringTab extends React.Component {
    render() {
        return(
            <Grid component='div' container spacing={3}>
                    <Grid item xs={12}>
                        <div></div>
                    </Grid>
                    <Grid item xs={8}>
                        <MakeScatter />
                    </Grid>
                    <Grid item xs={4}>
                        <h1> Metadata</h1>
                    </Grid>
                    <Grid item xs={12}>
                        <div></div>
                    </Grid>
            </Grid>
        )
    }
}
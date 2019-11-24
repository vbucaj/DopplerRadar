from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import copy
import joypy
from matplotlib import cm
from math import floor, ceil
import matplotlib.pyplot as plt
import pylab as plot
import itertools
import seaborn as sns


class DopplerRadar(object):

    def __init__(self, data_frame=None, drep_readings=[-0.3,0.1,0.4,0.75, 1,1.4,1.9,2.1,2.3,2.65,3,3.25,3.8], intv_width=float(0.25), number_samples=int(10)):
        # stats.gaussian_kde.__init__(self,data_frame,drep_readings,intv_width)

        self.data_frame = data_frame
        self.drep_readings = drep_readings
        self.intv_width = float(intv_width)
        self.number_samples = int(number_samples)

    def sample_drep_gep(self):
        sample_drep_dict = {}
        for i in self.drep_readings:
            j = self.drep_readings.index(i)
            x = np.array(self.data_frame[(self.data_frame[self.data_frame.columns[1]] < (i + self.intv_width)) & (
                    self.data_frame[self.data_frame.columns[1]] > (i - self.intv_width))][self.data_frame.columns[0]])
            my_kde = gaussian_kde(x)
            sample = my_kde.resample(self.number_samples)
            sample = sample.tolist()[0]
            sample_drep_dict[j] = sample

        sample_gep_lst = []
        for k in range(self.number_samples):
            sample = []
            for i in range(len(self.drep_readings)):
                sample.append(sample_drep_dict[i][k])
            sample_gep_lst.append(sample)
        return sample_drep_dict, sample_gep_lst

    def get_kde(self):
        x_val=[]
        my_kde_val=[]
        for i in self.drep_readings:
            j = self.drep_readings.index(i)
            x = np.array(self.data_frame[(self.data_frame[self.data_frame.columns[1]] < (i + self.intv_width)) & (
                    self.data_frame[self.data_frame.columns[1]] > (i - self.intv_width))][
                             self.data_frame.columns[0]])
            my_kde = gaussian_kde(x)
            x_val.append(x)
            my_kde_val.append(my_kde)
        return x_val, my_kde_val

    def get_percentiles(self):
        GEP_simulations,_=self.sample_drep_gep()

        per85 = []
        per50 = []
        per25 = []
        dict_len = len(GEP_simulations)
        for i in range(dict_len):
            GEP_simulations[i].sort()
            index85 = floor(len(GEP_simulations[i]) * 0.85)
            index50 = floor(len(GEP_simulations[i]) * 0.5)
            index25 = floor(len(GEP_simulations[i]) * 0.25)
            per85.append(GEP_simulations[i][index85])
            per50.append(GEP_simulations[i][index50])
            per25.append(GEP_simulations[i][index25])
        df_percent = pd.DataFrame([per25, per50, per85], columns=self.drep_readings, index=['25th Percentile', '50th Percentile', '85th Percentile'])
        return df_percent, per85, per50, per25


    def plot_percentiles(self, per25=None, per50=None,per85=None,figsize=(12,8),xlabel="DREP Daily Readings (StDev)",
                         ylabel="GEP Realizations (StDev)", xlabel_fontsize=int(14),ylabel_fontsize=int(14), title=None,
                         title_fontsize=int(18),legend_location='upper right', label=None,s=2000, lw=3, gep_marker='o',tick_param_labelsize=12,
                         marker_85='^',marker_50='s', marker_25='D',color_85='r',color_50='b',color_25='y',legend_fontsize=16, gep_markersize=int(8)):

        self.per25=per25
        self.per50=per50
        self.per85=per85
        #self.Generated_GEP_strings=Generated_GEP_strings
        self.figsize=figsize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlabel_fontsize = xlabel_fontsize
        self.ylabel_fontsize = ylabel_fontsize
        self.title_fontsize = title_fontsize
        self.label = label
        self.lw = lw
        self.gep_marker = gep_marker
        self.marker_85=marker_85
        self.marker_25=marker_25
        self.marker_50=marker_50
        self.color_85=color_85
        self.color_50=color_50
        self.color_25=color_25
        self.gep_markersize = gep_markersize
        self.s=s
        self.legend_location=legend_location
        self.legend_fontsize=legend_fontsize
        self.tick_param_labelsize=tick_param_labelsize

        _,Generated_GEP_strings=self.sample_drep_gep()

        plt.figure(figsize=self.figsize)
        for i in range(self.number_samples):
            plt.scatter(self.drep_readings, Generated_GEP_strings[i],label=None, marker=self.gep_marker, s=self.gep_markersize)
            plt.xlabel(xlabel=self.xlabel, fontsize=self.xlabel_fontsize)
            plt.ylabel(ylabel=self.ylabel, fontsize=self.ylabel_fontsize)
            #plt.title(title=self.title, fontsize=self.title_fontsize)

        plt.scatter(self.drep_readings, self.per85, marker=self.marker_85, s=self.s, color=self.color_85, lw=self.lw, label='85th Percentile')
        plt.scatter(self.drep_readings, self.per50, marker=self.marker_50, s=self.s, color=self.color_50, lw=self.lw, label='50th Percentile')
        plt.scatter(self.drep_readings, self.per25, marker=self.marker_25, s=self.s, color=self.color_25, lw=self.lw, label='25th Percentile')
        plt.xticks(self.drep_readings)
        plt.legend(loc=self.legend_location, fontsize=self.legend_fontsize)
        plt.tick_params(labelsize=self.tick_param_labelsize)

    def get_maxGEP(self):

        sample_drep_dict, sample_gep_lst=self.sample_drep_gep()
        Max_gep = []
        for item in sample_gep_lst:
            Max_gep.append(max(item))
        return Max_gep

    def plot_maxGEP_Dist(self,figsize=(8,6),domain=np.linspace(-1.5,6,1000),title=None,title_fontsize=14,alpha=0.3, axis_on_off='on',color='red', tick_parm_labelsize=14):

        self.figsize=figsize
        self.domain=domain
        self.alpha=alpha
        self.color=color
        self.tick_param_labelsize=tick_parm_labelsize
        self.axis_on_off=axis_on_off
        self.title=title
        self.title_fontsize=title_fontsize

        Max_gep=self.get_maxGEP()

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title(self.title, fontsize=self.title_fontsize)
        my_kde = gaussian_kde(Max_gep)
        ax.plot(self.domain, my_kde(self.domain))
        ax.fill_between(self.domain, my_kde(self.domain), alpha=self.alpha, color=self.color)
        ax.plot(Max_gep, np.full_like(Max_gep, -0.01), '|k', markeredgewidth=1)
        ax.axis(self.axis_on_off)
        plt.tick_params(labelsize=self.tick_param_labelsize)

class DopplerSensitivityAnalysis(object):

    def __init__(self,data_frame=None):


        self.data_frame=data_frame

    def plot_GepDist_DiscScale(self, DiscScale_list=[0.05,0.1,0.125,0.25],drep_pt=2.5,num_rows=1,num_columns=4, x_d=np.linspace(-2,7,1000),
                               alpha=0.5, markeredgewidth=1, figsize=(12,4), legend_loc=1, frameon=False, legend_fontsize=9,legend_handle_length=0.2):
        self.DiscScale_list = DiscScale_list
        self.drep_pt = drep_pt
        self.num_rows=num_rows
        self.num_columns=num_columns
        self.x_d=x_d
        self.alpha=alpha
        self.markeredgewidth=markeredgewidth
        self.legend_loc=legend_loc
        self.figsize=figsize
        self.frameon=frameon
        self.legend_fontsize=legend_fontsize
        self.legend_handle_length=legend_handle_length

        sub_gep = {}
        for j,i in enumerate(self.DiscScale_list):
            x = np.array(self.data_frame[(self.data_frame[self.data_frame.columns[1]] < ( self.drep_pt+i)) & (self.data_frame[self.data_frame.columns[1]] > (self.drep_pt-i))][self.data_frame.columns[0]])
            sub_gep[j]=x


        fig, ax = plt.subplots(self.num_rows, self.num_columns, figsize=self.figsize)


        color = iter(cm.rainbow(np.linspace(0, 1, len(self.DiscScale_list))))

        for j in range(self.num_columns):
            c = next(color)
            # ax[i,j].hist(sub_gep[ct],bins=30, label="Width {}".format(width_lst[ct]), density=True, color=c, alpha=0.5)
            mean=np.round(sub_gep[j].mean(),3)
            stdev=np.round(sub_gep[j].std(),3)
            n=len(sub_gep[j])
            my_kde = gaussian_kde(sub_gep[j])
            ax[j].plot(self.x_d, my_kde(self.x_d), color='k')
            ax[j].fill_between(self.x_d, my_kde(self.x_d), alpha=self.alpha, color=c, label="$\epsilon$= {} \n Mean={}\n StDev={}\n n= {} ".format(self.DiscScale_list[j], mean,stdev,n))
            ax[j].plot(sub_gep[j], np.full_like(sub_gep[j], -0.01), '|k', markeredgewidth=self.markeredgewidth)
            ax[j].legend(loc=self.legend_loc,frameon=self.frameon)
            ax[j].title.set_text("DREP reading: {}".format(self.drep_pt))
        params = {'legend.fontsize': self.legend_fontsize,'legend.handlelength': self.legend_handle_length}
        plot.rcParams.update(params)



class Runoff(DopplerRadar):
    def __init__(self,data_frame=None, drep_readings=[-0.3,0.1,0.4,0.75, 1,1.4,1.9,2.1,2.3,2.65,3,3.25,3.8], intv_width=float(0.25), number_samples=int(10),
                 CN=60,gauge_mean=23.88,gauge_stdev=28.19):
        DopplerRadar.__init__(self,data_frame, drep_readings, intv_width, number_samples)
        #self.data_frame = data_frame
        #self.Generated_GEP = Generated_GEP
        self.CN = CN
        #self.drep_readings=drep_readings
        self.gauge_mean=gauge_mean
        self.gauge_stdev=gauge_stdev



    def compute_runoff(self):

        _, Generated_GEP = self.sample_drep_gep()
        sorted_gep = copy.deepcopy(Generated_GEP)
        for item in sorted_gep:
            item.sort()
        P = [(np.multiply(i, self.gauge_stdev) + self.gauge_mean)/25.4 for i in sorted_gep]

        for lst in P:
            for i, val in enumerate(lst):
                lst[i] = max(val, 0)
        S = np.full_like(P, ((1000 / self.CN) - 10))  # fills array S with same length as P above
        Q = (np.square(P - 0.2 * S)) / (P + 0.8 * S)  # SCS run off formula

        return P, Q


    def plot_runoff(self,x_axis=12,y_axis=8,xlabel="Time in Days", ylabel="Run Off Estimates (cm)",xlabel_fontsize=int(14),
                    ylabel_fontsize=int(14), title="DualPol Run Off Estimates Over a Thirteen Day Period",
                    title_fontsize=int(22),label=None, lw=float(0.5), marker='o', markersize=int(8)):

        self.x_axis=x_axis
        self.y_axis=y_axis
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.title=title
        self.xlabel_fontsize=xlabel_fontsize
        self.ylabel_fontsize=ylabel_fontsize
        self.title_fontsize=title_fontsize
        self.label=label
        self.lw=lw
        self.marker=marker
        self.markersize=markersize

        _, Generated_GEP = self.sample_drep_gep()

        _, Q = self.compute_runoff()

        drep_inch=[(np.multiply(i,self.gauge_stdev) + self.gauge_mean)/25.4 for i in self.drep_readings]
        drep_inch=[round(i,2) for i in drep_inch]

        plt.figure(figsize=(self.x_axis, self.y_axis))
        for i in range(len(Generated_GEP)):
            plt.plot(drep_inch, Q[i], label=self.label, lw=self.lw, marker=self.marker, markersize=self.markersize)
            plt.xlabel(xlabel=self.xlabel, fontsize=self.xlabel_fontsize)
            plt.ylabel(ylabel=self.ylabel, fontsize=self.ylabel_fontsize)
            plt.title(label=self.title, fontsize=self.title_fontsize)
        plt.xticks(drep_inch)

    def plot_RunoffvsGEP(self,figsize=(8,6), N=1,marker='o',markerfacecolor='r', xlabel='GEP (in)', ylabel='Runoff (in)',
                         xlabel_size=14,ylabel_size=14, title=None,title_size=18, tick_param_size=12):

        self.figsize=figsize
        self.N=N
        self.marker=marker
        self.markerfacecolor=markerfacecolor
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.xlabel_size=xlabel_size
        self.ylabel_size=ylabel_size
        self.title=title
        self.title_size=title_size
        self.tick_param_size=tick_param_size

        P, Q = self.compute_runoff()

        plt.figure(figsize=self.figsize)
        for i in range(N):
            plt.scatter(P[i], Q[i], marker=self.marker)
            plt.xlabel(self.xlabel, fontsize=self.xlabel_size)
            plt.ylabel(self.ylabel, fontsize=self.ylabel_size)
            plt.title(self.title, fontsize=self.title_size)

        plt.tick_params(labelsize=self.tick_param_size)


class DopplerJoyPlot():

    def __init__(self, data_frame=None, drep_readings=[-0.3,0.1,0.4,0.75, 1,1.4,1.9,2.1,2.3,2.65,3,3.25,3.8],intv_width=float(0.125),kind="kde",
                      range_style='own', tails=0.2, yrot=22, ylabelsize=14, xlabelsize=14, linecolor='k',
                      overlap=5, linewidth=2, colormap=cm.gist_rainbow, fade=False, legend=False,
                      labels=[0.1,0.2].sort(), grid='both', figsize=(10, 10), alpha=0.5):
        self.data_frame = data_frame
        self.drep_readings = drep_readings
        self.intv_width=intv_width
        self.kind=kind
        self.range_style=range_style
        self.tails=tails
        self.yrot=yrot
        self.ylabelsize=ylabelsize
        self.xlabelsize=xlabelsize
        self.linecolor=linecolor
        self.overlap=overlap
        self.linewidth=linewidth
        self.colormap=colormap
        self.fade=fade
        self.legend=legend
        self.labels=labels
        self.grid=grid
        self.figsize=figsize
        self.alpha=alpha

    def data_frame_for_joyplot(self):
        df_lst=[]

        for i in self.drep_readings:
            x = list(self.data_frame[(self.data_frame[self.data_frame.columns[1]] < (i + self.intv_width)) & (
                    self.data_frame[self.data_frame.columns[1]] > (i - self.intv_width))][
                             self.data_frame.columns[0]])
            df = pd.DataFrame(np.array([[i for k in x], x]).T, columns=[self.data_frame.columns[1],self.data_frame.columns[0]])
            df_lst.append(df)
        return pd.concat(df_lst)


    def display_joy_plot(self):
        df_lst = []

        for i in self.drep_readings:
            x = list(self.data_frame[(self.data_frame[self.data_frame.columns[1]] < (i + self.intv_width)) & (
                    self.data_frame[self.data_frame.columns[1]] > (i - self.intv_width))][
                         self.data_frame.columns[0]])
            df = pd.DataFrame(np.array([[i for k in x], x]).T,
                              columns=[self.data_frame.columns[1], self.data_frame.columns[0]])
            df_lst.append(df)

        merged=pd.concat(df_lst)

        return joypy.joyplot(data=merged, by=self.data_frame.columns[1], column=self.data_frame.columns[0],
                             kind=self.kind,
                             range_style=self.range_style, tails=self.tails, yrot=self.yrot, ylabelsize=self.ylabelsize,
                             xlabelsize=self.xlabelsize, linecolor=self.linecolor,
                             overlap=self.overlap, linewidth=self.linewidth, colormap=self.colormap, fade=self.fade,
                             legend=self.legend,
                             labels=self.labels, grid=self.grid, figsize=self.figsize, alpha=self.alpha)









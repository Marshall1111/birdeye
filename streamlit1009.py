from io import StringIO
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np

st.set_page_config(
    page_title="解析txt",
    page_icon=":chart_with_upwards_trend:", #Emoji Shortcode（快捷代码）
    # layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

st.title(":chart_with_upwards_trend:"+"数据解析")
st.write('目前可以实现的功能：上传txt文件，自动解析障碍物和车道线数据，并提供csv下载。并绘制鸟瞰动图。')

uploaded_file = st.file_uploader("上传txt文件", type=".txt")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def read_txt(lines):
    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # # To read file as string:
    # lines = stringio.readlines()

    obs=[]
    lanes=[]

    for line in lines:
        line_list = line.strip("\n").split()
        if len(line_list) == 6:
            j3_frame_id = line_list[0]
            j3_ts = line_list[1]
            parsing_id = line_list[2]
            parsing_ts = line_list[3]
            parsing_time = line_list[4]
            type = line_list[5]
            head = [j3_frame_id,j3_ts,parsing_id,parsing_ts,parsing_time,type]
        elif len(line_list) == 0:
            head = []
        elif len(line_list) == 13:
            obs_line = head + line_list
            obs.append(obs_line)
        elif len(line_list) == 17:
            lane_line = head + line_list
            lanes.append(lane_line)

    df_lanes = pd.DataFrame(lanes)
    df_obs = pd.DataFrame(obs)

    lanes_col = ['j3_frame_id','j3_ts','parsing_id','parsing_ts','parsing_time','type','num','counter_id','original_id'
        ,'life_time_ms','lane_type','pos','conf','c0','c1','c2','c3','len','color','mark','x','y','z']
    obs_col = ['j3_frame_id','j3_ts','parsing_id','parsing_ts','parsing_time','type','num','counter_id','original_id'
               ,'obs_type','life_time_ms','age','cur_lane','pos_x','pos_y','pos_z','vel_x','vel_y','vel_z']

    df_lanes.columns = lanes_col
    df_obs.columns = obs_col

    return df_lanes,df_obs
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def jiexi(df_lanes,df_obs):
    obs_frame_min=min(df_obs["j3_frame_id"])
    obs_frame_max=max(df_obs['j3_frame_id'])
    ts_begin_obs = int(min(df_obs["parsing_time"]))
    time_begin_obs = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(ts_begin_obs/1000))
    ts_end_obs = int(max(df_obs["parsing_time"]))
    time_end_obs = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(ts_end_obs/1000))
    delta_obs = datetime.strptime(time_end_obs, "%Y-%m-%d %H:%M:%S")-datetime.strptime(time_begin_obs, "%Y-%m-%d %H:%M:%S")
    delta_s_obs = delta_obs.seconds

    lans_frame_min = min(df_lanes["j3_frame_id"])
    lans_frame_max = max(df_lanes['j3_frame_id'])
    ts_begin_lanes = int(min(df_lanes["parsing_time"]))
    time_begin_lanes = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(ts_begin_lanes/1000))
    ts_end_lanes = int(max(df_lanes["parsing_time"]))
    time_end_lanes = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(ts_end_lanes/1000))
    delta_lanes = datetime.strptime(time_end_lanes, "%Y-%m-%d %H:%M:%S")-datetime.strptime(time_begin_lanes, "%Y-%m-%d %H:%M:%S")
    delta_s_lanes = delta_lanes.seconds


    return obs_frame_min,obs_frame_max,lans_frame_min,lans_frame_max,time_begin_obs,time_end_obs,delta_s_obs,time_begin_lanes,time_end_lanes,delta_s_lanes
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def plot_bird_eye(dataframe_obs,dataframe_lans,frame_min=0,frame_max=999999999):

    dataframe_obs = dataframe_obs[(dataframe_obs['j3_frame_id'].astype('int')>=frame_min) &(dataframe_obs['j3_frame_id'].astype('int')<=frame_max)]
    dataframe_obs = dataframe_obs.sort_values(by=['j3_frame_id','original_id','obs_type'],axis=0,ascending=[True,False,True])
    dataframe_lans = dataframe_lans[(dataframe_lans['j3_frame_id'].astype('int') >= frame_min) & (dataframe_lans['j3_frame_id'].astype('int') <= frame_max)]
    # st.write(dataframe_lans)

    # obs表中的Frames
    frames = dataframe_obs['j3_frame_id'].dropna().unique().tolist()
    frames = [str(i) for i in frames]

    # 障碍物类型
    types = dataframe_obs['obs_type'].dropna().unique().tolist()

    # make figure 创建图表字典，整个图表就是一个字典，里面包含data、layout、frames三个字段
    # 其中layout是一个字典，而data和frames都是列表
    # 参考参数 https://plotly.com/python/reference/scatter/#scatter
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # 填充图表的layout字段，fill in most of layout
    # xy轴的范围
    fig_dict["layout"]["xaxis"] = dict(range=[15, -15],
                                       title="Lat Distance")
    fig_dict["layout"]["yaxis"] = dict(range=[0, 100],
                                       title="Long Distance")
    # 鼠标悬浮展示的模式，closest是靠近的时候显示悬浮信息
    fig_dict["layout"]["hovermode"] = "closest"
    # 更新菜单
    fig_dict["layout"]["updatemenus"] = [
        dict(
                buttons=[
                    dict(args=[None,
                               dict(frame=dict(duration=20
                                               ,redraw=False
                                               ),
                                    fromcurrent=True,
                                    transition=dict(duration=0
                                                    # ,easing="quadratic-in-out"
                                                    )
                                    )
                                ],
                         label="Play",
                         method="animate"
                         ),
                    dict(args=[[None],
                               dict(frame=dict(duration=0
                                               ,redraw=False
                                               ),
                                    mode="immediate",
                                    transition=dict(duration=0
                                                    # ,easing="quadratic-in-out"
                                                    )
                                    )
                                ],
                         label="Pause",
                         method="animate"
                         )
                ],
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                type="buttons",
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ]
    # 滑块的字典,设置延时是0
    sliders_dict = dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=20),
            prefix="Frameid:",
            visible=True,
            xanchor="right",
        ),
        transition=dict(duration=0
                        # ,easing="cubic-in-out"
                        ),
        pad={"b": 10, "t": 50},
        len=0.9,
        x=0.1,
        y=0,
        steps=[]
    )

    # 对不同的障碍物种类可以分配不同的颜色和大小
    colormap = {'0':'red','1':'green','9':'blue','18':'MediumSlateBlue'}
    sizemap = {'0':100,'1':200,'9':300,'18':150}

    # make data
    # 第一帧的图片
    frameid = frames[0]

    # 分不同的障碍物种类绘制散点-----------
    for type in ['0','1','9','18']:
        dataset_by_frameid = dataframe_obs[dataframe_obs["j3_frame_id"].astype('int') == int(float(frameid))]
        dataset_by_frameid_and_type = dataset_by_frameid[dataset_by_frameid["obs_type"] == type]

        if dataset_by_frameid_and_type.shape[0] > 0:
            data_dict = dict(
                x=list(dataset_by_frameid_and_type["pos_y"].astype('float')),
                y=list(dataset_by_frameid_and_type["pos_x"].astype('float')),
                mode="markers+text",
                text=[str(int(i)) for i in dataset_by_frameid_and_type["original_id"]],
                textposition="top center",
                # 在动画中，要指定一下ids,同样的original_id会被认为是同一个对象，否则会发生位置交替现象
                ids=list(dataset_by_frameid_and_type["original_id"]),
                marker=dict(
                    sizemode="area",
                    size=[sizemap[i] for i in dataset_by_frameid_and_type["obs_type"]],
                    color=[colormap[i] for i in dataset_by_frameid_and_type["obs_type"]],
                    opacity=0.5,
                ),
                name=type,
                # 不显示图例
                showlegend=False,
                # 悬停显示的文本
                hovertext=["id:" + str(int(i)) + '<br>' + 'balabala' for i in
                           dataset_by_frameid_and_type['original_id']],
            )
        else:
            data_dict = dict(x=[0], y=[0], mode="markers",marker=dict(size=0,opacity=0.0),showlegend=False,)
        # 每一种类别，都作为图表字典data字段中的一个元素
        fig_dict["data"].append(data_dict)

    # 画车道线------------
    dataset_by_frameid_lans = dataframe_lans[dataframe_lans["j3_frame_id"].astype('int') == int(float(frameid))].reset_index()
    # 每根车道线画一个,动画的时候，后面帧中的车道线的个数会始终与第一帧的个数相等，所以从第一帧就画10根车道线，如果不足10根，则画空的
    for i in range(10):
        if i<=max(dataset_by_frameid_lans.index):
    # for i in dataset_by_frameid_lans.index:
            x0 = float(dataset_by_frameid_lans.iloc[i]["x"])
            len = float(dataset_by_frameid_lans.iloc[i]["len"])
            c0 = float(dataset_by_frameid_lans.iloc[i]["c0"])
            c1 = float(dataset_by_frameid_lans.iloc[i]["c1"])
            c2 = float(dataset_by_frameid_lans.iloc[i]["c2"])
            c3 = float(dataset_by_frameid_lans.iloc[i]["c3"])
            id = dataset_by_frameid_lans.iloc[i]["original_id"]
            x_list = np.linspace(x0, x0+len, 50)
            y_list = [c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 for x in x_list]
            data_dict_lane = dict(x=y_list,
                                  y=x_list,
                                  name=str(int(id)),
                                  mode="lines",
                                  hovertext = str(int(id)), #悬停显示的文本
                                  hoverinfo = "name", #"x", "y", "x+y", "x+y+z", "all"
                                  line=dict(shape="spline",
                                            color="darkslategray",
                                            ),
                                  showlegend=False
                                  )
        else:
            # pass
            data_dict_lane = dict(x=[0], y=[0], mode="markers",marker=dict(size=0,opacity=0.0),showlegend=False,)
        # 每一跟车道线，都作为图表字典data字段中的一个元素
        fig_dict["data"].append(data_dict_lane)

    # 画速度方向
    # st.write(dataset_by_frameid)
    dataset_by_frameid = dataset_by_frameid.reset_index()
    for i in range(20):
        if i<=max(dataset_by_frameid.index):
    # for i in dataset_by_frameid.index:
            x0=float(dataset_by_frameid.iloc[i]["pos_x"])
            v_x=float(dataset_by_frameid.iloc[i]["vel_x"])
            x1=x0+float(dataset_by_frameid.iloc[i]["vel_x"])*5
            y0=float(dataset_by_frameid.iloc[i]["pos_y"])
            v_y = float(dataset_by_frameid.iloc[i]["vel_y"])
            y1=y0+float(dataset_by_frameid.iloc[i]["vel_y"])*5
            data_dict_vec = dict(x=[y0, y1],
                                 y=[x0, x1],
                                 mode="lines",
                                 hovertext='velocity:('+str(round(v_x,2))+","+str(round(v_y,2))+")m/s",  # 悬停显示的文本
                                 hoverinfo="text",  # "x", "y", "x+y", "x+y+z", "all"
                                 line=dict(shape="spline",
                                           dash="dot",
                                           color='magenta'
                                           ),
                                 showlegend=False
                                 )
        else:
            data_dict_vec=dict(x=[0], y=[0], mode="markers",marker=dict(size=0,opacity=0.0),showlegend=False,)
        fig_dict["data"].append(data_dict_vec)

    # 创建帧make frames
    for frameid in frames:
        frame = dict(data=[]
                     ,name=frameid
                     )
        dataset_by_frameid = dataframe_obs[dataframe_obs["j3_frame_id"].astype('int') == int(float(frameid))].reset_index()
        # -----------分不同的障碍物种类绘制散点-----------
        for type in ['0','1','9','18']:
            dataset_by_frameid_and_type = dataset_by_frameid[dataset_by_frameid["obs_type"] == type]
            if dataset_by_frameid_and_type.shape[0]>0:
                data_dict = dict(
                    x=list(dataset_by_frameid_and_type["pos_y"].astype('float')),
                    y=list(dataset_by_frameid_and_type["pos_x"].astype('float')),
                    mode="markers+text",
                    text=[str(int(i)) for i in dataset_by_frameid_and_type["original_id"]],
                    textposition="top center",
                    # 在动画中，要指定一下ids,同样的original_id会被认为是同一个对象，否则会发生位置交替现象
                    ids=list(dataset_by_frameid_and_type["original_id"]),
                    marker=dict(
                        sizemode="area",
                        size=[sizemap[i] for i in dataset_by_frameid_and_type["obs_type"]],
                        color=[colormap[i] for i in dataset_by_frameid_and_type["obs_type"]],
                        opacity=0.5,
                    ),
                    name=type,
                    # 不显示图例
                    showlegend=False,
                    # 悬停显示的文本
                    hovertext=["id:" + str(int(i)) + '<br>' + 'balabala' for i in dataset_by_frameid_and_type['original_id']],
                )
            else:
                data_dict = dict(x=[0], y=[0], mode="markers",marker=dict(size=0,opacity=0.0),showlegend=False,)
            # 每一个类别，都是这个frame里面，data字段中的一个元素
            frame["data"].append(data_dict)

        # ------------画车道线------------
        dataset_by_frameid_lans = dataframe_lans[dataframe_lans["j3_frame_id"].astype('int') == int(float(frameid))].reset_index()
        # dataset_by_frameid_lans = dataframe_lans[dataframe_lans["j3_frame_id"] == int(float(frameid))]

        # 每根车道线画一个
        for i in range(10):
            if i<=max(dataset_by_frameid_lans.index):
        # for i in dataset_by_frameid_lans.index:
                x0 = float(dataset_by_frameid_lans.iloc[i]["x"])
                len = float(dataset_by_frameid_lans.iloc[i]["len"])
                c0 = float(dataset_by_frameid_lans.iloc[i]["c0"])
                c1 = float(dataset_by_frameid_lans.iloc[i]["c1"])
                c2 = float(dataset_by_frameid_lans.iloc[i]["c2"])
                c3 = float(dataset_by_frameid_lans.iloc[i]["c3"])
                id = dataset_by_frameid_lans.iloc[i]["original_id"]
                x_list = np.linspace(x0, x0 + len, 50)
                y_list = [c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 for x in x_list]
                data_dict_lane = dict(x=y_list,
                                      y=x_list,
                                      name=str(int(i)),
                                      mode="lines",
                                      hovertext=str(int(i)),  # 悬停显示的文本
                                      hoverinfo="name",  # "x", "y", "x+y", "x+y+z", "all"
                                      line=dict(shape="spline",
                                                color="darkslategray",
                                                ),
                                      showlegend=False
                                      )
            else:
                # pass
                data_dict_lane = dict(x=[0], y=[0], mode="markers",marker=dict(size=0,opacity=0.0),showlegend=False,)
            # 每一根车道线，都作为图表字典data字段中的一个元素
            frame["data"].append(data_dict_lane)


        # -----画速度方向----------------
        # dataset_by_frameid = dataframe_obs[dataframe_obs["j3_frame_id"] == int(float(frameid))]
        for i in range(20):
            if i <= max(dataset_by_frameid.index):
                # print(i)
                # print(dataset_by_frameid)
                # for i in dataset_by_frameid.index:
                x0 = float(dataset_by_frameid.iloc[i]["pos_x"])
                v_x = float(dataset_by_frameid.iloc[i]["vel_x"])
                x1 = x0 + float(dataset_by_frameid.iloc[i]["vel_x"]) * 5
                y0 = float(dataset_by_frameid.iloc[i]["pos_y"])
                v_y = float(dataset_by_frameid.iloc[i]["vel_y"])
                y1 = y0 + float(dataset_by_frameid.iloc[i]["vel_y"]) * 5
                data_dict_vec = dict(x=[y0, y1],
                                     y=[x0, x1],
                                     mode="lines",
                                     hovertext='velocity:(' + str(round(v_x, 2)) + "," + str(round(v_y, 2)) + ")m/s",
                                     # 悬停显示的文本
                                     hoverinfo="text",  # "x", "y", "x+y", "x+y+z", "all"
                                     line=dict(shape="spline",
                                               dash="dot",
                                               color='magenta'
                                               ),
                                     showlegend=False
                                     )
            else:
                data_dict_vec = dict(x=[0], y=[0], mode="markers",marker=dict(size=0,opacity=0.0),showlegend=False,)
            frame["data"].append(data_dict_vec)


        # ----------------------------
        # 每一帧，都是图表字典frames字段中的一个元素
        fig_dict["frames"].append(frame)
        slider_step = dict(
            args=[
                [frameid],
                dict(
                    frame={"duration": 0, "redraw": False},
                    mode="immediate",
                    transition={"duration": 0}
                )
            ],
            label=str(int(float(frameid))),
            method="animate"
        )
        sliders_dict["steps"].append(slider_step)


    # 把前面创建的滑块的字典，包一层列表，放在layout下面的slider字段中
    fig_dict["layout"]["sliders"] = [sliders_dict]
    # 根据前面的图表字典创建图表对象fig
    fig = go.Figure(fig_dict)
    return fig
data=0
# If CSV is not uploaded and checkbox is filled, use values from the example file
# and pass them down to the next if block
if uploaded_file is None:
    if use_example_file:
        with open('example.txt') as file_object:
            # line = file_object.readline()  # 读取一行;指针自动下移
            lines = file_object.readlines()  # 读取每一行存在一个列表中
        df_lanes, df_obs = read_txt(lines)
        data=1
        # st.write(df_lanes)

if uploaded_file is not None:
    # read_txt(uploaded_file)
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    lines = stringio.readlines()
    df_lanes,df_obs = read_txt(lines)
    data=1


if data==1:
    obs_frame_min,obs_frame_max,lans_frame_min,lans_frame_max,time_begin_obs,time_end_obs,delta_s_obs,time_begin_lanes,time_end_lanes,delta_s_lanes=jiexi(df_lanes,df_obs)
    st.write("障碍物最小帧:{}，最大帧:{}。开始时间：{}，结束时间：{}。持续时间：{}秒".format(obs_frame_min,obs_frame_max,time_begin_obs,time_end_obs,delta_s_obs))
    # 转换格式后下载
    csv_obs = convert_df(df_obs)
    st.download_button(
        label="下载障碍物csv数据",
        data=csv_obs,
        file_name='obs.csv',
        mime='text/csv',
    )

    st.write("车道线最小帧:{}，最大帧:{}。开始时间：{}，结束时间：{}。持续时间：{}秒".format(obs_frame_min,obs_frame_max,time_begin_lanes,time_end_lanes,delta_s_lanes))
    # 转换格式后下载
    csv_lanes = convert_df(df_lanes)
    st.download_button(
        label="下载车道线csv数据",
        data=csv_lanes,
        file_name='lanes.csv',
        mime='text/csv',
    )
    col1, col2, col3= st.columns(3)
    frame_min=int(obs_frame_min)
    frame_max=int(obs_frame_max)
    with col1:
        frame_min_str = st.text_input('最小帧', obs_frame_min)
    with col2:
        frame_max_str = st.text_input('最大帧', obs_frame_max)

    plot = st.button('区间绘图')
    if plot:
        frame_min = int(frame_min_str)
        frame_max = int(frame_max_str)

    reset = st.button('整体绘图')
    if reset:
        frame_min = int(obs_frame_min)
        frame_max = int(obs_frame_max)

    fig = plot_bird_eye(df_obs,df_lanes,frame_min=frame_min,frame_max=frame_max)
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        # paper_bgcolor="LightSteelBlue",
        # autosize = False,
        # width = 800,
        height=800,
    )
    st.plotly_chart(fig, use_container_width=False)
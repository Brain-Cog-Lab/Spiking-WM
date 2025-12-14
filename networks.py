import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as td
import node
import normalization
from einops import rearrange
import tools


class RSSM(nn.Module):
    def __init__(
        self,
        config,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act="LIFNode",
        norm="PopNorm",
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="MCRNN",
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        act_p = eval("config."+act) 
        norm_p = eval("config."+norm)
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device

        self.T = config.spike_times

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, **norm_p))
            inp_layers.append(act(**act_p))
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(tools.weight_init)

        self._cell = eval(cell)(config, self._hidden, self._deter, **config.dyn_cell_p)
        self._cell.apply(tools.weight_init)
        # if cell == "gru":
        #     self._cell = GRUCell(self._hidden, self._deter)
        #     self._cell.apply(tools.weight_init)
        # elif cell == "gru_layer_norm":
        #     self._cell = GRUCell(self._hidden, self._deter, norm=True)
        #     self._cell.apply(tools.weight_init)
        # else:
        #     raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, **norm_p))
            img_out_layers.append(act(**act_p))
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, **norm_p))
            obs_out_layers.append(act(**act_p))
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.weight_init)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.weight_init)

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

 
    def initial(self, batch_size):
        # deter = torch.zeros(batch_size, self._deter).to(self._device)
        deter = torch.zeros(self.T, batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        # TODO
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(self.T, batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        embed = rearrange(embed, "st b t d -> t st b d") 
        # print("$"*100)
        # print("embed in observation shape: ", embed.shape)
        action, is_first = tools.swap(action), tools.swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (time, batch, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        # deter [times, spike_time, batch, deter_num] -> [spike_time, batch, time, deter_num]
        post = {k: tools.swap(v, (0, 1, 2), (1, 2, 0)) if k=="deter" else tools.swap(v) for k, v in post.items()}
        prior = {k: tools.swap(v, (0, 1, 2), (1, 2, 0)) if k=="deter" else tools.swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = tools.swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: tools.swap(v) for k, v in prior.items()}
        return prior
    

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        stoch= torch.unsqueeze(stoch, 0)
        re_shape =  [1] * len(stoch.shape)
        re_shape[0] = state["deter"].shape[0]  # spike_times
        return torch.cat([stoch.repeat(*re_shape), state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = td.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                td.independent.Independent(td.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        self.reset()
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        # print("#"*100)
        # print("#"*100)
        # print("prev_state deter:  ", prev_state["deter"].shape)
        # 初始化 is_firest 对应 idx的 state, 更新prev_state内容
        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                if key == "deter":
                    is_first_r = torch.reshape(
                        is_first, 
                        (1,) + is_first.shape + (1,) * (len(val.shape) -1 - len(is_first.shape))
                    )
                else:
                    is_first_r = torch.reshape(
                        is_first,
                        is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                    )
                # print("#"*100)
                # print("key: ", key)
                # print("prev_state[key] shape: ", prev_state[key].shape)
                # print("val shape: ", val.shape)
                # print("is_first_r shape: ", is_first_r.shape)
                # print("init_state[key] shape: ", init_state[key].shape)
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )
        # print("prev_state deter:  ", prev_state["deter"].shape)
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                # print("#"*100)
                # print("deter shape: ", prior["deter"].shape)
                # print("stoch shape: ", prior["stoch"].shape)
                # print("embed shape: ", embed.shape)
                x = torch.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            xs = []
            for step in range(self.T):
                xs.append(self._obs_out_layers(x[step]))
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            xs = sum(xs) /self.T
            stats = self._suff_stats_layer("obs", xs)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    # this is used for making future image

    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # (batch, stoch, discrete_num)
        self.reset()
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros([self.T] + shape)
            # TODO 这里没用被执行
            # embed [s_t, bacth, dimes]
            # (batch, stoch * discrete_num) -> (self.T, batch, stoch * discrete_num + action, embed)
            x = torch.cat([prev_stoch.expand(self.T, *list(prev_stoch.shape)), 
                           prev_action.expand(self.T, *list(prev_action.shape)),
                            embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
            # x = x.expand(self.T, *list(x.shape))
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        # 需要变成 spike
        x_spikes = []
        if self._shared:
            for step in range(self.T):
                x_spikes.append(self._inp_layers(x[step]))
        else:
            for step in range(self.T):
                x_spikes.append(self._inp_layers(x))
        deter = prev_state["deter"]   
        # 这里 rec_depth 被设置成1
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            # deter 为[x_spikes]
            x_spikes, deter = self._cell(x_spikes, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        xs = []
        for step in range(self.T):
            xs.append(self._img_out_layers(x_spikes[step]))
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        xs = sum(xs) / self.T
        # TODO 
        stats = self._suff_stats_layer("ims", xs)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior


    def get_stoch(self, deter):
        self.reset()
        spikes = []
        for step in range(self.T):
            x = self._img_out_layers(deter[step])
            spikes.append(x)
        x = sum(spikes) / self.T
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()


    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = td.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        config,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = SpikeConvEncoder(
                input_shape, cnn_depth, act, eval("config."+act), norm, eval("config."+norm), kernel_size, minres, config.spike_times
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = SpikeMLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                config[act],
                norm,
                config[norm],
                symlog_inputs=symlog_inputs,
                spike_times=config.spike_times
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            cnn_out = self._cnn(inputs) # cnn output spike list []
            outputs.append(torch.stack(cnn_out, dim=0))   
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            mlp_out = self._mlp(inputs, unroll=True)  # mlp out spike list []
            outputs.append(torch.stack(mlp_out, dim=0))

        outputs = torch.cat(outputs, -1)

        return outputs



class MultiDecoder(nn.Module):
    def __init__(
        self,
        config,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = SpikeConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                eval("config."+act),
                norm,
                eval("config."+norm),
                kernel_size,
                minres,
                cnn_sigmoid=cnn_sigmoid,
                spike_times=config.spike_times
            )
        if self.mlp_shapes:
            self._mlp = SpikeMLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                eval("config."+act),
                norm,
                eval("config."+norm),
                vector_dist,
                spike_times=config.spike_times
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                td.independent.Independent(td.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class SpikeConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="LIFNode",
        act_p = {},
        norm="PopNorm",
        norm_p = {},
        kernel_size=4,
        minres=4,
        spike_times=5
    ):
        super().__init__()
        self.T = spike_times
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        h, w, input_ch = input_shape
        layers = []
        
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2**i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChNorm(out_dim, norm, norm_p))
            layers.append(act(**act_p))
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        self.reset()
        outs =[]
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)

        inp = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        inp = inp.permute(0, 3, 1, 2)

        for step in range(self.T):
            # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
            # x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
            # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
            # x = x.permute(0, 3, 1, 2)
            x = self.layers(inp)
            # (batch * time, ...) -> (batch * time, -1)
            x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
            # (batch * time, -1) -> (batch, time, -1)
            outs.append(x.reshape(list(obs.shape[:-3]) + [x.shape[-1]]))
        return outs
    
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()



class SpikeConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act="LIF",
        act_p={},
        norm="PopNorm",
        norm_p = {},
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
        spike_times=5
    ):
        super(SpikeConvDecoder, self).__init__()
        self.T = spike_times
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres**2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools.weight_init)
        in_dim = self._embed_size // (minres**2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres**2) // (2 ** (i + 1))
            bias = False
            initializer = tools.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = tools.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ChNorm(out_dim, norm, norm_p))
            if act:
                layers.append(act(**act_p))
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        # TODO feature shape
        self.reset()
        spikes = []
        for step in range(self.T):
            x = self._linear_layer(features[step])
        # (batch, time, -1) -> (batch , time, h, w, ch)
            x = x.reshape(
                [-1, self._minres, self._minres, self._embed_size // self._minres**2]
            )
            # (batch, time, -1) -> (batch * time, ch, h, w)
            x = x.permute(0, 3, 1, 2)
            x = self.layers(x)
            spikes.append(x)
        mean = sum(spikes)/self.T
        # print("#"*100)
        # print("mean shape: ", mean.shape)
        # print("feature shape: ", features.shape)
        # print("self._shape: ", self._shape)
        # (batch,*time, ch, h, w) -> (batch, time, ch, h, w) necessary???
        mean = mean.reshape(features.shape[1:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean
    
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()


class SpikeMLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="LIF",
        act_p={},
        norm="PopNorm",
        norm_p={},
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        spike_times=5
    ):
        super(SpikeMLP, self).__init__()
        self.T = spike_times
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, **norm_p))
            layers.append(act(**act_p))
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None, unroll=False, pr=False):
        self.reset()
        if unroll:
            shape = [self.T] + [1] * len(features.shape)
            features = features.unsqueeze(0).repeat(*shape)
        if self._symlog_inputs:
            features = tools.symlog(features)
        outs = []
        if pr:
            print("features shape: ", features.shape)
        for step in range(self.T):
            outs.append(self.layers(features[step]))
        outs =  sum(outs) / self.T
        
        if self._shape is None:
            return outs
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](outs)
                if self._std == "learned":
                    std = self.std_layer[name](outs)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(outs)
            if self._std == "learned":
                std = self.std_layer(outs)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return tools.ContDist(
                td.independent.Independent(
                    td.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return tools.ContDist(
                td.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return tools.Bernoulli(
                td.independent.Independent(
                    td.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return tools.DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return tools.SymlogDist(mean)
        raise NotImplementedError(dist)
    
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()


class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act="LIF",
        act_p={},
        norm="PopNorm",
        norm_p={},
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
        spike_times=5
    ):
        super(ActionHead, self).__init__()
        self.T= spike_times
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(node, act)
        norm = getattr(normalization, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp


        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, **norm_p))
            pre_layers.append(act(**act_p))
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        # x = features
        self.reset()
        xs = []
        for step in range(self.T):
            s = self._pre_layers(features[step])
            xs.append(s)
        x = sum(xs) / self.T
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.normal.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = td.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = td.normal.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = td.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = td.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(td.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = td.normal.Normal(mean, 1)
            dist = tools.ContDist(td.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(td.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(td.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist

    def reset(self):
            for mod in self.modules():
                if hasattr(mod, 'n_reset'):
                    mod.n_reset()

class MCRNN(nn.Module): 
    def __init__(self, 
                 config,
                 input_size, 
                 size, 
                 act="MCNode",
                 norm="PopNorm",
                 device="cuda"):
        super().__init__()
        self._device = device
        self._norm = norm
        self.neuron = getattr(node, act)(**eval("config."+act))
        self._size = size
        self.apcial_w = nn.Linear(input_size+size, size)
        self.basal_w = nn.Linear(input_size+size, size) 
        self.soma_w = nn.Linear(input_size, size)
        # self.h2o = nn.Linear(size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1) 
        self.T = config.spike_times
        # TODO
        if norm:
            norm = getattr(normalization, norm)
            self.apical_norm=norm(size, **eval("config."+self._norm))
            self.basal_norm=norm(size, **eval("config."+self._norm))
            self.soma_norm=norm(size, **eval("config."+self._norm))
    

    def forward(self, inputs, state):
        # input spike list or tensor [spike_T, batch, ...]
        # state spike list or tensor [spike_T, batch, ...]
        self.reset()
        # combined = torch.cat((inputs, hidden), -1) # [T, batch, dims]
        # assert combined.shape[0] == self.T
        outputs = []
        # print("inputs device: ", inputs.device)
        for step in range(self.T):
            inp_spike = inputs[step]
            # print("#"*100)
            # print("state shape: ", state.shape)
            # print("inspike sahpe: ", inp_spike.shape)
            # print("state[step] shape: ", state[step].shape)
            combined = torch.cat((inp_spike, state[0][step]), -1)
            apcial_inp = self.apcial_w(combined)
            basal_inp = self.basal_w(combined)
            soma_inp = self.soma_w(inp_spike)
            if self._norm:
                apcial_inp = self.apical_norm(apcial_inp)
                basal_inp = self.basal_norm(basal_inp)
                soma_inp = self.soma_norm(soma_inp)
            neu_input= {"basal_inputs":basal_inp, "apical_inputs": apcial_inp, "soma_inputs": soma_inp}
            spike = self.neuron(neu_input)
            # print("spike shape: ", spike.shape)
            outputs.append(spike)
        # out
        # output = self.h2o(sum(outputs)/self.T)
        # output = self.softmax(output)
        return outputs, [torch.stack(outputs, dim=0)]

    def initHidden(self):
        return torch.zeros(self.T, 1, self.hidden_size).to(self._device)

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset() 



# class GRUCell(nn.Module):
#     def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
#         super(GRUCell, self).__init__()
#         self._inp_size = inp_size
#         self._size = size
#         self._act = act
#         self._norm = norm
#         self._update_bias = update_bias
#         self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
#         if norm:
#             self._norm = nn.LayerNorm(3 * size, eps=1e-03)

#     @property
#     def state_size(self):
#         return self._size

#     def forward(self, inputs, state):
#         state = state[0]  # Keras wraps the state in a list.
#         parts = self._layer(torch.cat([inputs, state], -1))
#         if self._norm:
#             parts = self._norm(parts)
#         reset, cand, update = torch.split(parts, [self._size] * 3, -1)
#         reset = torch.sigmoid(reset)
#         cand = self._act(reset * cand)
#         update = torch.sigmoid(update + self._update_bias)
#         output = update * cand + (1 - update) * state
#         return output, [output]


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret



class ChNorm(nn.Module):
    def __init__(self, ch, norm, norm_p):
        super(ChNorm, self).__init__()
        self.norm = norm(ch, **norm_p)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# 

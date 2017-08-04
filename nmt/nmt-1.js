/*var encode_time_step = function (src, model, lh, hidden_sizes, G, x=false)
{
  if (x)
    G = new R.Graph(false);
  /*else
    var G = new R.Graph();*/
  var inp = one_hot($vocab_sz_src, $vocab_src[src]);
  var out = R.forwardRNN(G, model, hidden_sizes, inp, lh);

  return [G, out];
}

var encode_update = function (model, G, dw)
{
  G.backward();
  var solver = new R.Solver();
  solver.step(model, 0.01, 0.0001, 5.0);
}


var decode_time_step = function (src, tgt, model, lh, solver, hidden_sizes, context, G)
{
  var inp = cat(one_hot($vocab_sz_tgt, $vocab_tgt[src]), context);
  var out = R.forwardRNN(G, model, hidden_sizes, inp, lh);
  var logprobs = out.o;
  var probs = R.softmax(logprobs);
  var target = $vocab_tgt[tgt];
  cost = -Math.log(probs.w[target]);
  logprobs.dw = probs.w;
  logprobs.dw[target] -= 1;

  return [model, cost, out, probs, G];
}

  var hidden_sizes = [20];
/*var train = function (hidden_sizes)
{*/
  var encoder = R.initRNN($vocab_sz_src, hidden_sizes, $vocab_sz_src+100);
  encoder["name"] = "encoder";
  var decoder = R.initRNN($vocab_sz_src+100+$vocab_sz_tgt, hidden_sizes, $vocab_sz_tgt);
  decoder["name"] = "decoder";

  decoder["encoder_Wxh0"] = encoder["Wxh0"];
  decoder["encoder_Whh0"] = encoder["Whh0"];
  decoder["encoder_bhh0"] = encoder["bhh0"];
  decoder["encoder_Whd"]  = encoder["Whd"];
  decoder["encoder_bd"]   = encoder["bd"];

  var solver = new R.Solver();
  var G = new R.Graph();


  lh = {};
  costs = [];
  var l = 0;
  while (true)
  {
    var cost = 0.0;
    for (var i=0; i<$data.length; i++) {
      var last_encode_lh,enc_g;
      for (var j=0; j<$data[i]["src"].length; j++) {
        [G, lh] = encode_time_step($data[i]["src"][j], encoder, lh, hidden_sizes, G);
      }
      last_encode_lh = lh;

      var context = last_encode_lh.o;
      lh = {}
      for (var k=0; k<$data[i]["tgt"].length; k++) {
        var [decoder, c, lh, probs, G] = decode_time_step($data[i]["tgt"][k], $data[i]["tgt"][k+1], decoder, lh, solver, hidden_sizes, context, G);
        cost += c;

        G.backward();
        solver.step(encoder, 0.01, 0.0001, 5.0);
        solver.step(decoder, 0.01, 0.0001, 5.0);
      }

      //encode_update(encoder, enc_g, lh);

    }
    l++;
    costs.push(cost);
    if (stopping_criterion(costs[costs.length-2], cost, l))
      break;
  }

  //return [encoder, decoder, costs];

/*}*/


//var x = ["<bos>", "das", "ist", "<eos>"];



var k = 3;
var kbest = [];

for (var q=0; q < k; q++) {

var x = $data[0]["src"]; 
var _, cntxt;
var o = {};
for (var i=0; i<x.length; i++)
  [_, o] = encode_time_step(x[i], encoder, o, hidden_sizes, null, true);
cntxt = o.o;

var w = "<bos>", lh;
var prev = {};
var str = "";
var z = 0;
var p = 0;
  while (true) {
    var G = new R.Graph(false);
    var inp = cat(one_hot($vocab_sz_tgt, $vocab_tgt[w]), cntxt);
    var lh = R.forwardRNN(G, decoder, hidden_sizes, inp, prev);
    prev = lh;
    var logprobs = lh.o;
    var probs = R.softmax(logprobs);
    var x = R.samplei(probs.w);
    p += probs.w[x];
    src = $ivocab_tgt[x];
    str += " "+src;
    z++;
    if (src == "<eos>" || z==100)
      break;
  }
  kbest.push( {"transl":str, "score":p } );
}



var main = function ()
{
  /*var hidden_sizes = [20];
  var [encoder, decoder, costs] = train($data, hidden_sizes);*/

  return false;
}

$(document).ready(function()
{
  main();
});*/


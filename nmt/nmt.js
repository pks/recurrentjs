/*var $data = [{"src":["<bos>", "das", "ist", "ein", "kleines", "haus", "<eos>"],
              "tgt":["<bos>", "this", "is", "a", "small", "house", "<eos>"]}
];*/

var $data = [{"src":["<bos>", "das", "<eos>"],
              "tgt":["<bos>", "this", "<eos>"]}];//,
/*             {"src":["<bos>", "mein", "<eos>"],
              "tgt":["<bos>", "my", "<eos>"]}];*/

var make_vocab = function (data)
{
  var k = 0;
  var vocab = {}, ivocab = [];
  for (var i=0; i<data.length; i++) {
    for (var j=0; j<data[i].length; j++) {
      var w = data[i][j];
      if (vocab[w]==undefined) {
        vocab[w] = k;
        ivocab[k] = w;
        k++;
      }
    }
  }

  return [vocab,ivocab,ivocab.length];
}

var $vocab_src, $ivocab_src,$vocab_sz_src;
var $vocab_tgt, $ivocab_tgt,$vocab_sz_tgt;
[$vocab_src,$ivocab_src,$vocab_sz_src] = make_vocab($data.map(function(i){return i["src"]}));
$vocab_sz_src++;
[$vocab_tgt,$ivocab_tgt,$vocab_sz_tgt] = make_vocab($data.map(function(i){return i["tgt"]}));

var one_hot = function (n, i)
{
  var m = new R.Mat(n,1);
  m.set(i, 0, 1);

  return m;
}

var stopping_criterion = function (c, d, iter, margin=0.01, max_iter=100)
{
  if (Math.abs(c-d) < margin || iter>=max_iter)
    return true;
  return false;
}

var cat = function (a, b)
{
  R.assert(a.d==1 && b.d==1);
  var m = new R.Mat(a.n+b.n, 1);
  var i;
  for (i=0; i<a.n; i++) {
    m.w[i] = a.w[i];
    m.dw[i] = a.dw[i];
  }
  for (var j=0; j<b.n; j++) {
    m.w[i+j] = b.w[j];
    m.dw[i+j] = b.dw[j];
  }

  return m;
}

function shuffle(array) {
  var currentIndex = array.length, temporaryValue, randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

var hidden_sizes = [5];
var encoder = R.initRNN($vocab_sz_src, hidden_sizes, $vocab_sz_src+100);
encoder["name"] = "encoder";
var decoder = R.initRNN($vocab_sz_src+100+$vocab_sz_tgt, hidden_sizes, $vocab_sz_tgt);
decoder["name"] = "decoder";

var solver = new R.Solver();
var h = {};
var inp = null;
costs = [];
var epoch = 0;

decoder["encoder_Wxh0"] = encoder["Wxh0"];
decoder["encoder_Whh0"] = encoder["Whh0"];
decoder["encoder_bhh0"] = encoder["bhh0"];
decoder["encoder_Whd"]  = encoder["Whd"];
decoder["encoder_bd"]   = encoder["bd"];

var x = encoder.Whh0.w[0]
var y = decoder.Whh0.w[0]

var enc_hs = [];
var dec_hs = [];
var dec_inps = [];

while (true)
{
  $data = shuffle($data);
  var c = 0.0;
  for (var i=0; i<$data.length; i++) {
    var last_enc_h;
    var G = new R.Graph();
    for (var j=0; j<$data[i]["src"].length; j++) {
      var src = $data[i]["src"][j];
      var inp = one_hot($vocab_sz_src, $vocab_src[src]);
      var h = R.forwardRNN(G, encoder, hidden_sizes, inp, h);
      last_enc_h = h;
      enc_hs.push(h);
    }

    for (var z=0; z<G.backprop.length; z++) {
      G.backprop[z]["mark"] = "encoder";
    }

    var context = last_enc_h.o;
    var h = {}
    for (var k=0; k<$data[i]["tgt"].length; k++) {
      var src = $data[i]["tgt"][k],
          tgt = $data[i]["tgt"][k+1];

      inp = cat(one_hot($vocab_sz_tgt, $vocab_tgt[src]), context);
      dec_inps.push(inp);
      h = R.forwardRNN(G, decoder, hidden_sizes, inp, h);
      dec_hs.push(h);
      
      var logprobs = h.o;
      var probs = R.softmax(logprobs);
      var target = $vocab_tgt[tgt];
      cost = -Math.log(probs.w[target]);
      if (!cost || cost==Infinity) cost = 0; // hmmm
      logprobs.dw = probs.w;
      logprobs.dw[target] -= 1;
      c += cost;

      // update weights
      G.backward();
      // copy grads? decoder.Wxh0 -> encoder.Whd ?
      for (var z=0; z<last_enc_h.o.dw.length; z++) {
        last_enc_h.o.dw[z] = dec_inps[0].dw[z];
      }
      G.backward1();
      //exit();
      solver.step(decoder, 0.01, 0.000001, 5.0);
    }
  }

  epoch++;
  costs.push(c);
  if (stopping_criterion(costs[costs.length-2], cost, epoch))
    break;
}



var k = 10;
var samples = [];

for (var q=0; q < k; q++) {

//var x = ["<bos>", "das", "ist", "ein", "kleines", "haus", "<eos>"]; 
var x = ["<bos>", "das", "<eos>"]; 
var _, cntxt;
var o = {};
var G = new R.Graph(false);
for (var i=0; i<x.length; i++) {
    var src = x[i];
    var inp = one_hot($vocab_sz_src, $vocab_src[src]);
    var h = R.forwardRNN(G, encoder, hidden_sizes, inp, h);
}

cntxt = h.o;

var w = "<bos>", lh;
var prev = {};
var str = "";
var z = 0;
var p = 1;
  while (true) {
    var inp = cat(one_hot($vocab_sz_tgt, $vocab_tgt[w]), cntxt);
    var lh = R.forwardRNN(G, decoder, hidden_sizes, inp, prev);
    prev = lh;
    var logprobs = lh.o;
    var probs = R.softmax(logprobs);
    var x = R.samplei(probs.w);
    p *= probs.w[x];
    src = $ivocab_tgt[x];
    str += " "+src;
    z++;
    if (src == "<eos>" || z==100)
      break;
  }
  samples.push( {"transl":str, "score":p } );
}

best = "";
best_score = -999999;
for (var q = 0; q<samples.length; q++) {
  if (best_score < samples[q].score) {
    best = samples[q].transl
  }
}

alert(best);


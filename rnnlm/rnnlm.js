var $data = [["<bos>", "this", "is", "my", "house", "<eos>"],
            ["<bos>", "welcome", "to", "my", "house", "<eos>"],
            ["<bos>", "welcome", "to", "my", "tiny", "house", "<eos>"],
            ["<bos>", "welcome", "to", "my", "little", "house", "<eos>"]];

var make_vocab = function ($data)
{
  var k = 0;
  var vocab = {}, ivocab = [];
  for (var i=0; i<$data.length; i++) {
    for (var j=0; j<$data[i].length; j++) {
      var w = $data[i][j];
      if (vocab[w]==undefined) {
        vocab[w] = k;
        ivocab[k] = w;
        k++;
      }
    }
  }

  return [vocab,ivocab,ivocab.length];
}

var $vocab, $ivocab,$vocab_sz;
[$vocab,$ivocab,$vocab_sz] = make_vocab($data);

var one_hot = function (n, i)
{
  var m = new R.Mat(n,1);
  m.set(i, 0, 1);

  return m;
}

var time_step = function (src, tgt, model, lh, solver, hidden_sizes, G)
{
  var inp = one_hot($vocab_sz, $vocab[src]);
  var out = R.forwardLSTM(G, model, hidden_sizes, inp, lh);
  var logprobs = out.o;
  var probs = R.softmax(logprobs);
  var target = $vocab[tgt];
  cost = -Math.log(probs.w[target]);
  logprobs.dw = probs.w;
  logprobs.dw[target] -= 1;
  G.backward();
  solver.step(model, 0.01, 0.0001, 5.0);

  return [model, cost, out, probs];
}

var stopping_criterion = function (c, d, iter, margin=0.01, max_iter=100)
{
  if (Math.abs(c-d) < margin || iter>=max_iter)
    return true;
  return false;
}

var train = function ($data, hidden_sizes)
{
  var model = R.initLSTM($vocab_sz, hidden_sizes, $vocab_sz);
  var solver = new R.Solver();
  lh = {};
  costs = [];
  var k = 0;
  while (true)
  {
    $data = shuffle($data);
    var cost = 0.0;
    for (var i=0; i<$data.length; i++) {
      var G = new R.Graph();
      for (var j=0; j<$data[i].length-1; j++) {
        [model, c, lh, probs] = time_step($data[i][j], $data[i][j+1], model, lh, solver, hidden_sizes, G);
        cost += c;
      }
    }
    k++;
    costs.push(cost);
    if (stopping_criterion(costs[costs.length-2], cost, k))
      break;
  }

  return [model, costs];
}

var generate = function (model, hidden_sizes)
{
  var prev = {};
  var str = "<bos>";
  var src = "<bos>";
  while (true) {
    var G = new R.Graph(false);
    var inp = one_hot($vocab_sz, $vocab[src]);
    var lh = R.forwardLSTM(G, model, hidden_sizes, inp, prev);
    prev = lh;
    var logprobs = lh.o;
    var probs = R.softmax(logprobs);
    var x = R.samplei(probs.w);
    src = $ivocab[x];
    str += " "+src;
    if (src == "<eos>")
      break;
  }

  return str;
}

var predict = function (model, hidden_sizes, context)
{
  var prev = {}, lh;
  var logprobs, probs;
  for (var i=0; i<context.length; i++) {
    var G = new R.Graph(false);
    var inp = one_hot($vocab_sz, $vocab[context[i]]);
    lh = R.forwardLSTM(G, model, hidden_sizes, inp, prev);
    prev = lh;
    logprobs = lh.o;
    probs = R.softmax(logprobs);
  }

  var maxi = R.maxi(probs.w);

  return [$ivocab[maxi], probs.w[maxi], probs.w];
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

var main = function ()
{
  var hidden_sizes = [10];
  var [model, costs] = train($data, hidden_sizes);

  for (var i=0; i<costs.length; i++)
    $("#costs").append('<li>'+costs[i]+'</li>');

  var i=0;
  while (1) {
    var s = generate(model, hidden_sizes);
    $("#samples").append("<li>"+s+"</li>");
    i++;
    if (i==13) break;
  }

  var context = ["<bos>"];
  var [t,tp,dist] = predict(model, hidden_sizes, context);
  $("#predict0").html(t+" ("+tp+")");

  context = ["<bos>", "welcome"];
  var [t,tp,dist] = predict(model, hidden_sizes, context);
  $("#predict1").html(t+" ("+tp+")");

  context = ["<bos>", "welcome", "to"];
  var [t,tp,dist] = predict(model, hidden_sizes, context);
  $("#predict2").html(t+" ("+tp+")");

  context = ["<bos>", "welcome", "to", "my"];
  var [t,tp,dist] = predict(model, hidden_sizes, context);
  $("#predict3").html(t+" ("+tp+")");

  context = ["<bos>", "welcome", "to", "my", "house"];
  var [t,tp,dist] = predict(model, hidden_sizes, context);
  $("#predict4").html(t+" ("+tp+")");

  return false;
}

$(document).ready(function()
{
  main();
});


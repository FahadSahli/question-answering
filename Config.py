from argparse import ArgumentParser
import json

class Config:

    @classmethod
    def arguments_parser(cls):
        parser = ArgumentParser()

        parser.add_argument("--data_type", dest="data_type", required=False)
        parser.add_argument("-d", "--data", dest="data_dir", help="path to pre-processed data", required=True)
        parser.add_argument("-w", "--weight_path", dest="weight_path", help="path to save and load models", required=False)
        
        parser.add_argument("-e", "--embedding_file", dest="embedding_file", help="path to embeddings file", required=True)
        parser.add_argument("--embedding_dim", dest="embedding_dim", required=False)
        
        parser.add_argument("--batch_size", dest="batch_size", required=False)
        parser.add_argument("--num_epoches", dest="num_epoches", required=False)
        
        parser.add_argument("--optimizer", dest="optimizer", required=False)
        parser.add_argument("--learning_rate", dest="learning_rate", required=False)
        parser.add_argument("--beta1", dest="beta1", required=False)
        parser.add_argument("--beta2", dest="beta2", required=False)
        parser.add_argument("--grad_clipping", dest="grad_clipping", required=False)

        parser.add_argument("--hidden_size", dest="hidden_size", required=False)
        parser.add_argument("--dropout_rate", dest="dropout_rate", required=False)
        parser.add_argument("--two_encoding_layers", dest="two_encoding_layers", required=False)

        parser.add_argument("--training", dest="training", required=False)
        parser.add_argument("--testing", dest="testing", required=False)
        return parser

    def __init__(self):
        args = self.arguments_parser().parse_args()

        self.data_type = args.data_type if args.data_type else "NE"
        self.data_dir = args.data_dir
        self.weight_path = args.weight_path if args.weight_path else "./"

        self.embedding_file = args.embedding_file
        self.embedding_dim = int(args.embedding_dim)  if args.embedding_dim else 200

        self.batch_size = int(args.batch_size)  if args.batch_size else 64
        self.num_epoches = int(args.num_epoches)  if args.num_epoches else 100

        self.optimizer = args.optimizer  if args.optimizer else "ADAM"
        self.learning_rate = float(args.learning_rate)  if args.learning_rate else 0.001
        self.beta1 = float(args.beta1)  if args.beta1 else 0.9
        self.beta2 = float(args.beta2)  if args.beta2 else 0.999
        self.grad_clipping = float(args.grad_clipping)  if args.grad_clipping else 10

        self.hidden_size = int(args.hidden_size)  if args.hidden_size else 384
        self.dropout_rate = float(args.dropout_rate)  if args.dropout_rate else 0.0
        self.two_encoding_layers = json.loads(args.two_encoding_layers.lower())  if args.two_encoding_layers else False

        self.training = json.loads(args.training.lower())  if args.training else True
        self.testing = json.loads(args.testing.lower())  if args.testing else False
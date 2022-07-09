#include <bits/stdc++.h>
#include <boost/random.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

typedef unsigned long long uLL;
typedef long long LL;

class MetaPathSampler{
private:
    vector< vector<int> > E;
    vector< vector<char> edge_type;
    int num_node, num_threads;
    long long num_edge;
    int nw, wl;
    int *ret_path, *ret_dis;

public:
    MetaPathSampler(np::ndarray edge_index_np, LL batch_size, LL num_walk, LL walk_len, LL num_workers);
    ~MetaPathSampler() {
        delete ret_path;
        delete ret_dis;
    }
    inline void create_sample_thread(LL *batch, LL L, LL R);
    p::object sample(np::ndarray batch);
};

const char* init() {
    Py_Initialize();
    np::initialize();
    cerr << "Initialize meta-path sampler." << endl;
    return "MetaPathSampler.init()";
}


MetaPathSampler::MetaPathSampler(np::ndarray edge_index_np, LL batch_size, LL num_walk, LL walk_len, LL num_workers) {
    // Check "edge_index"
    LL nd_edge_index_np = edge_index_np.get_nd();
    if (nd_edge_index_np != 2 || edge_index_np.shape(0) != 3)
        throw std::runtime_error("\"edge_index\" must be 2-dimensional numpy.ndarray shapes like [3, num_edge]. ");
    if (edge_index_np.get_dtype() != np::dtype::get_builtin<LL>())
        throw std::runtime_error("\"edge_index\" must be int64 numpy.ndarray. ");

    num_node = 0;
    num_edge = edge_index_np.shape(1);
    nw = num_walk;
    wl = walk_len;

    LL *edge_index = reinterpret_cast<LL *>(edge_index_np.get_data());
    for (LL i=0;i<2*num_edge;i++) {
        if (edge_index[i] < 0) {
            throw std::runtime_error("Negative node_id in \"edge_index\". ");
        }
        num_node = max(num_node, edge_index[i]+1);
    }

    E.resize(num_node);
    edge_type.resize(num_node);
    for (LL i=0;i<num_edge;i++) {
        int u = (int)edge_index[i];
        int v = (int)edge_index[num_edge+i];
        char t = (char)edge_index[2*num_edge+i];
        E[u].push_back(v);
        edge_type[u].push_back(t);
    }

    ret_path = new int[batch_size*nw*wl];
    ret_dis = new int[batch_size*nw*wl];
    num_threads = num_workers;

    cerr << "MetaPathSampler(#V: " << num_node << ", #E: " << num_edge << "). " << endl;
}

inline void MetaPathSampler::create_sample_thread(LL *batch, LL L, LL R) {
    for (LL st=L;st<R;st++) {
        for (LL i=0;i<nw;i++) {
            LL u = batch[st];
            LL t = 0;
            for (LL _=0;_<wl;_++) {
                LL pos = st*(nw*wl)+i*wl+_;
                ret_path[pos] = u;
                ret_dis[pos] = t;
                int num_nei = (int)E[u].size();
                int tmp = rand() % num_nei;
                t = edge_type[u][tmp];
                u = E[u][tmp];
            }
        }
    }
    return ;
}

p::object MetaPathSampler::sample(np::ndarray batch_np) {
    int nd_batch_np = batch_np.get_nd();
    if (nd_batch_np != 1)
        throw std::runtime_error("\"edge_index\" must be 1-dimensional numpy.ndarray shapes like [num_node]. ");
    if (batch_np.get_dtype() != np::dtype::get_builtin<LL>())
        throw std::runtime_error("\"batch\" must be int64 numpy.ndarray. ");
    std::vector<std::thread> thread_pool_;

    LL *batch = reinterpret_cast<LL *>(batch_np.get_data());
    LL batch_size = batch_np.shape(0);
    LL num_block_node = batch_size / num_threads + 1;
    for (LL i=0;i<num_threads;i++) {
        LL L = i * num_block_node;
        LL R = (i+1) * num_block_node;
        R = min(R, batch_size);
        thread_pool_.push_back( std::thread( &MetaPathSampler::create_sample_thread, this, batch, L, R) );
    }
    for (LL i=0;i<(LL)thread_pool_.size();i++) thread_pool_[i].join();

    np::dtype dt = np::dtype::get_builtin<int>();
    p::tuple shape = p::make_tuple(batch_size, nw, wl);
    p::tuple stride = p::make_tuple(sizeof(int)*nw*wl, sizeof(int)*wl, sizeof(int));
    p::object own;
    np::ndarray ret_path_np = np::from_data(ret_path, dt, shape, stride, own);
    np::ndarray ret_dis_np = np::from_data(ret_dis, dt, shape, stride, own);
    return p::make_tuple(ret_path_np, ret_dis_np);
}

BOOST_PYTHON_MODULE(path_sampler) {
    boost::python::def("init", init);
    boost::python::class_<MetaPathSampler>("MetaPathSampler", boost::python::init<np::ndarray, LL, LL, LL, LL>())
        .def("sample", &MetaPathSampler::sample)
    ;
}
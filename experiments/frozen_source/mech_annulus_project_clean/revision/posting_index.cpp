// TYPE-1: exact Hamming-ball lookup over occupied-code tries and norm postings.
// Query memory is proportional to retrieved postings, never to reference n.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>
namespace py=pybind11;
using Clock=std::chrono::steady_clock;
using i64=long long;
static i64 elapsed(Clock::time_point t){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count();}
struct Node {int child[2]={-1,-1};std::vector<int> buckets;};
struct Bucket {uint64_t code;std::map<int,std::vector<int>> layers;};
struct Table {std::vector<Node> trie;std::vector<Bucket> buckets;};
class PostingIndex {
 int bits_;std::vector<int> layers_;std::vector<int> zero_;std::vector<Table> tables_;std::map<int,int> norm_sizes_;
 void search(const Table& t,int node,int bit,uint64_t q,int allowance,std::vector<int>& out,i64& visits)const{
  if(node<0||allowance<0)return;++visits;
  if(bit<0||allowance>=bit+1){out.insert(out.end(),t.trie[node].buckets.begin(),t.trie[node].buckets.end());return;}
  int v=(q>>bit)&1;search(t,t.trie[node].child[v],bit-1,q,allowance,out,visits);search(t,t.trie[node].child[1-v],bit-1,q,allowance-1,out,visits);
 }
public:
 PostingIndex(const std::vector<std::vector<uint64_t>>& codes,const std::vector<int>& layers,const std::vector<int>& zeros,int bits):bits_(bits),layers_(layers),zero_(zeros){
  if(bits<1||bits>32)throw std::invalid_argument("unsupported bit count");
  for(int l:layers_)norm_sizes_[l]++;
  for(const auto& v:codes){
   if(v.size()!=layers_.size())throw std::invalid_argument("reference length mismatch");
   Table t;std::map<uint64_t,int> occupied;
   for(size_t i=0;i<v.size();++i){auto it=occupied.find(v[i]);int k;
    if(it==occupied.end()){k=t.buckets.size();occupied[v[i]]=k;t.buckets.push_back(Bucket{v[i],{}});}else k=it->second;
    t.buckets[k].layers[layers_[i]].push_back(i);
   }
   t.trie.emplace_back();
   for(size_t k=0;k<t.buckets.size();++k){int node=0;t.trie[node].buckets.push_back(k);
    for(int bit=bits_-1;bit>=0;--bit){int v=(t.buckets[k].code>>bit)&1;
     if(t.trie[node].child[v]<0){int next=t.trie.size();t.trie[node].child[v]=next;t.trie.emplace_back();}
     node=t.trie[node].child[v];t.trie[node].buckets.push_back(k);
    }
   }tables_.push_back(std::move(t));
  }
 }
 py::dict query(const std::vector<uint64_t>& qcodes,double a,const std::vector<double>& radii,double delta,int hits,const std::vector<int>& thresholds)const{
  auto total_start=Clock::now();
  if(qcodes.size()!=tables_.size()||radii.size()!=thresholds.size()||radii.empty()||delta<=0)throw std::invalid_argument("query shape");
  if(!std::is_sorted(radii.begin(),radii.end())||!std::is_sorted(thresholds.begin(),thresholds.end()))throw std::invalid_argument("non-nested rule");
  for(int t:thresholds)if(t<0||t>bits_)throw std::invalid_argument("threshold range");
  hits=std::min(std::max(hits,1),int(tables_.size()));
  i64 norm_ns=0,hash_ns=0,collect_ns=0,intersection_ns=0,dedup_ns=0,annulus_ns=0,visits=0,bucket_visits=0,posting_ids=0,norm_postings=0;
  auto ts=Clock::now();
  int outer_low=int(std::floor(std::max(0.,a-radii.back())/delta));int outer_high=int(std::ceil((a+radii.back())/delta));
  int outer_size=0;for(auto it=norm_sizes_.lower_bound(outer_low);it!=norm_sizes_.end()&&it->first<=outer_high;++it){outer_size+=it->second;norm_postings+=it->second;}
  norm_ns+=elapsed(ts);
  std::map<int,std::vector<int>> cache;
  for(int threshold:thresholds){
   if(cache.count(threshold))continue;
   std::vector<int> retrieved;
   for(size_t l=0;l<tables_.size();++l){const auto& table=tables_[l];std::vector<int> buckets;
    ts=Clock::now();search(table,0,bits_-1,qcodes[l],threshold,buckets,visits);hash_ns+=elapsed(ts);bucket_visits+=buckets.size();
    for(int k:buckets){const auto& norms=table.buckets[k].layers;
     ts=Clock::now();auto first=norms.lower_bound(outer_low);auto last=norms.upper_bound(outer_high);norm_ns+=elapsed(ts);
     ts=Clock::now();for(auto it=first;it!=last;++it){posting_ids+=it->second.size();retrieved.insert(retrieved.end(),it->second.begin(),it->second.end());}collect_ns+=elapsed(ts);
    }
   }
   // One point occurs at most once per table: multiplicity is the table-hit count.
   ts=Clock::now();std::sort(retrieved.begin(),retrieved.end());std::vector<int> accepted;
   for(size_t i=0;i<retrieved.size();){size_t j=i+1;while(j<retrieved.size()&&retrieved[j]==retrieved[i])++j;if(int(j-i)>=hits)accepted.push_back(retrieved[i]);i=j;}
   cache.emplace(threshold,std::move(accepted));dedup_ns+=elapsed(ts);
  }
  std::vector<std::vector<int>> rings,balls;std::vector<int> previous;
  for(size_t j=0;j<radii.size();++j){
   ts=Clock::now();int low=int(std::floor(std::max(0.,a-radii[j])/delta));int high=int(std::ceil((a+radii[j])/delta));norm_ns+=elapsed(ts);
   ts=Clock::now();std::vector<int> current;
   for(int id:cache.at(thresholds[j]))if(layers_[id]>=low&&layers_[id]<=high&&!std::binary_search(zero_.begin(),zero_.end(),id))current.push_back(id);
   if(a<=radii[j]){current.insert(current.end(),zero_.begin(),zero_.end());std::sort(current.begin(),current.end());}
   intersection_ns+=elapsed(ts);
   ts=Clock::now();std::vector<int> ring;std::set_difference(current.begin(),current.end(),previous.begin(),previous.end(),std::back_inserter(ring));
   if(!std::includes(current.begin(),current.end(),previous.begin(),previous.end()))throw std::runtime_error("nesting failure");
   rings.push_back(std::move(ring));previous=std::move(current);annulus_ns+=elapsed(ts);
  }
  py::dict stats;stats["norm_lookup_ns"]=norm_ns;stats["hash_lookup_ns"]=hash_ns;stats["candidate_collect_ns"]=collect_ns;stats["candidate_intersection_ns"]=intersection_ns;
  stats["dedup_ns"]=dedup_ns;stats["annulus_build_ns"]=annulus_ns;stats["hash_bucket_visits"]=bucket_visits;stats["retrieved_posting_ids"]=posting_ids;stats["norm_posting_ids"]=norm_postings;stats["code_trie_node_visits"]=visits;
  stats["hash_id_checks"]=0;stats["code_distance_checks"]=0;stats["posting_core_ns"]=elapsed(total_start);
  py::dict out;out["union"]=previous;out["rings"]=rings;out["stats"]=stats;out["outer_norm_pool_size"]=outer_size;return out;
 }
};
PYBIND11_MODULE(_posting_index,m){py::class_<PostingIndex>(m,"PostingIndex").def(py::init<const std::vector<std::vector<uint64_t>>&,const std::vector<int>&,const std::vector<int>&,int>()).def("query",&PostingIndex::query);}

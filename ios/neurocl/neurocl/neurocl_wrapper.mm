//
//  neurocl_wrapper.m
//  neurocl
//
//  Created by Albert Murienne on 14/03/2017.
//  Copyright © 2017 Blackccpie. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "neurocl_wrapper.h"
#import "neurocl.h"

@interface NeuroclWrapper ()

@property (nonatomic, readonly) std::shared_ptr<neurocl::network_manager_interface> net_manager;

@end

@implementation NeuroclWrapper

- (instancetype) init
{
    return [self initWithNet:@"topology.txt" weights:@"weights.bin"];
}

- (instancetype) initWithNet:(NSString*) topology weights:(NSString*) weights
{
    self = [super init];
    
    if(self != nil)
    {
        std::string str_topology([topology UTF8String]);
        std::string str_weights([weights UTF8String]);
        
        _net_manager = neurocl::network_factory::build();
        _net_manager->load_network( str_topology, str_weights );
    }
    
    return self;
}

- (void) dealloc
{
    _net_manager.reset();
}

- (NSString*) digit_recognizer:(UIImage*) in
{
    return @"TODO";
}

@end
